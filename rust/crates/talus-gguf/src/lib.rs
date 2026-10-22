//! GGUF v2/v3 mmap reader, split-shard combiner, Strata native pack reader,
//! MTP runtime pack reader, draft vocabulary reader, and shape-catalog validator.

pub mod draft_vocab;
pub mod error;
pub mod gguf;
pub mod mmap;
pub mod mtp_rt;
pub mod pack;
pub mod shapes;
pub mod split;

pub use draft_vocab::DraftVocabFile;
pub use error::{GgufError, Result};
pub use gguf::{
    block_geometry, compute_tensor_byte_size, quant_type_from_name, GgufFile, GgufValue,
    TensorEntry, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_MAX_DIMS,
};
pub use mmap::{Backing, Mmap};
pub use mtp_rt::{
    MtpDenseTensor, MtpRtPack, MtpTensorKind, MTP_EXPERT_BLOB_BYTES, MTP_REQUIRED_DENSE_TENSORS,
};
pub use pack::{
    expert_row_bytes, ExpertLayerSpec, ExpertLayout, ExpertLayoutConfig, IndexRow, NativeFmt,
    PackFileId, PackIndex, StrataPack, WeightKind, DEFAULT_EXPERT_N_EMBD, DEFAULT_EXPERT_N_EXPERT,
    DEFAULT_EXPERT_N_FF, DEFAULT_EXPERT_N_LAYERS, NATIVE_EXPERTS_MAX_VERSION,
    NVFP4_EXPERT_TAIL_BYTES,
};
pub use shapes::{
    validate_matrix_shapes_dir, CatalogTensor, CatalogValidationSummary, ComponentCatalog,
    ShapeCatalog, MATRIX_SHAPE_CATALOG_FILES,
};
pub use split::{gguf_split_paths, GgufModel, GgufShards, TensorLocation};
pub use talus_core_api::{DraftVocab, GgufView, QuantType};

/// Crate identifier constant.
pub const CRATE_NAME: &str = "talus-gguf";

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(1);

    /// Creates an isolated scratch directory under the workspace's `target/test-scratch/talus-gguf-tests`
    /// (never under `/tmp`).
    struct ScratchDir {
        path: PathBuf,
    }

    impl ScratchDir {
        fn new(label: &str) -> Self {
            let id = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../target/test-scratch"
            ))
            .join("talus-gguf-tests")
            .join(format!("{label}-{pid}-{id}"));
            fs::create_dir_all(&path).expect("create scratch dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-gguf");
    }

    // ------------------------------------------------------------------------
    // Synthetic GGUF binary builder (mirroring tests/core/gguf_fixture.hpp)
    // ------------------------------------------------------------------------

    #[derive(Clone)]
    struct SyntheticTensor {
        name: String,
        dims: Vec<u64>,
        qtype: QuantType,
        seed: u8,
    }

    fn pattern_byte(seed: u8, idx: usize) -> u8 {
        seed.wrapping_mul(131)
            .wrapping_add(((idx * 17) ^ (idx >> 5)) as u8)
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn push_value(buf: &mut Vec<u8>, val: &GgufValue) {
        match val {
            GgufValue::U8(v) => {
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.push(*v);
            }
            GgufValue::I8(v) => {
                buf.extend_from_slice(&1u32.to_le_bytes());
                buf.push(*v as u8);
            }
            GgufValue::U16(v) => {
                buf.extend_from_slice(&2u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I16(v) => {
                buf.extend_from_slice(&3u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::U32(v) => {
                buf.extend_from_slice(&4u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I32(v) => {
                buf.extend_from_slice(&5u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F32(v) => {
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::Bool(v) => {
                buf.extend_from_slice(&7u32.to_le_bytes());
                buf.push(u8::from(*v));
            }
            GgufValue::String(s) => {
                buf.extend_from_slice(&8u32.to_le_bytes());
                push_str(buf, s);
            }
            GgufValue::Array(items) => {
                buf.extend_from_slice(&9u32.to_le_bytes());
                let elem_type = match items.first() {
                    Some(GgufValue::U32(_)) | None => 4u32,
                    Some(GgufValue::I32(_)) => 5u32,
                    Some(GgufValue::String(_)) => 8u32,
                    Some(GgufValue::F32(_)) => 6u32,
                    _ => 4u32,
                };
                buf.extend_from_slice(&elem_type.to_le_bytes());
                buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    match item {
                        GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
                        GgufValue::String(s) => push_str(buf, s),
                        _ => unreachable!(),
                    }
                }
            }
            GgufValue::U64(v) => {
                buf.extend_from_slice(&10u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::I64(v) => {
                buf.extend_from_slice(&11u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            GgufValue::F64(v) => {
                buf.extend_from_slice(&12u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    fn build_gguf_bytes(
        version: u32,
        kvs: &[(&str, GgufValue)],
        tensors: &[SyntheticTensor],
        truncate_tail: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());

        let mut alignment = GGUF_DEFAULT_ALIGNMENT as usize;
        for (k, v) in kvs {
            if *k == "general.alignment" {
                if let Some(a) = v.as_u64() {
                    if a > 0 && a.is_power_of_two() {
                        alignment = a as usize;
                    }
                }
            }
            push_str(&mut buf, k);
            push_value(&mut buf, v);
        }

        let mut offsets = Vec::with_capacity(tensors.len());
        let mut sizes = Vec::with_capacity(tensors.len());
        let mut cur_off = 0u64;
        for t in tensors {
            let sz = compute_tensor_byte_size(t.qtype, &t.dims, &None, &t.name).unwrap();
            offsets.push(cur_off);
            sizes.push(sz as usize);
            let align_u64 = alignment as u64;
            cur_off = (cur_off + sz).div_ceil(align_u64) * align_u64;
        }

        for (i, t) in tensors.iter().enumerate() {
            push_str(&mut buf, &t.name);
            buf.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for &d in &t.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&t.qtype.ggml_type_id().to_le_bytes());
            buf.extend_from_slice(&offsets[i].to_le_bytes());
        }

        if !tensors.is_empty() {
            let pad = (alignment - (buf.len() % alignment)) % alignment;
            buf.resize(buf.len() + pad, 0);
            for (i, t) in tensors.iter().enumerate() {
                let sz = sizes[i];
                for b_idx in 0..sz {
                    buf.push(pattern_byte(t.seed, b_idx));
                }
                if i + 1 < tensors.len() {
                    let pad = (alignment - (sz % alignment)) % alignment;
                    buf.resize(buf.len() + pad, 0);
                }
            }
        }

        if truncate_tail > 0 && truncate_tail < buf.len() {
            buf.truncate(buf.len() - truncate_tail);
        }
        buf
    }

    // ------------------------------------------------------------------------
    // 1. GgufFile single-file v2 & v3 + mmap + all metadata types + errors
    // ------------------------------------------------------------------------

    #[test]
    fn gguf_v2_and_v3_all_value_types_and_mmap() {
        let dir = ScratchDir::new("gguf-single");
        let kvs = vec![
            (
                "general.architecture",
                GgufValue::String("qwen4exp".to_string()),
            ),
            ("general.alignment", GgufValue::U32(64)),
            ("test.u8", GgufValue::U8(250)),
            ("test.i8", GgufValue::I8(-42)),
            ("test.u16", GgufValue::U16(65000)),
            ("test.i16", GgufValue::I16(-12345)),
            ("test.u32", GgufValue::U32(12345678)),
            ("test.i32", GgufValue::I32(-87654321)),
            ("test.f32", GgufValue::F32(3.5)),
            ("test.bool_true", GgufValue::Bool(true)),
            ("test.bool_false", GgufValue::Bool(false)),
            ("test.u64", GgufValue::U64(1 << 40)),
            ("test.i64", GgufValue::I64(-(1i64 << 40))),
            ("test.f64", GgufValue::F64(1.23456789)),
            (
                "test.arr_i32",
                GgufValue::Array(vec![
                    GgufValue::I32(11),
                    GgufValue::I32(11),
                    GgufValue::I32(10),
                ]),
            ),
            (
                "test.arr_str",
                GgufValue::Array(vec![
                    GgufValue::String("tok0".to_string()),
                    GgufValue::String("tok1".to_string()),
                ]),
            ),
        ];
        let tensors = vec![
            SyntheticTensor {
                name: "token_embd.weight".to_string(),
                dims: vec![64, 16],
                qtype: QuantType::Q8_0,
                seed: 1,
            },
            SyntheticTensor {
                name: "blk.0.attn_norm.weight".to_string(),
                dims: vec![64],
                qtype: QuantType::F32,
                seed: 2,
            },
            SyntheticTensor {
                name: "blk.0.ffn_gate_exps.weight".to_string(),
                dims: vec![256, 4, 2],
                qtype: QuantType::IQ4_XS,
                seed: 3,
            },
        ];

        for version in [2u32, 3u32] {
            let bytes = build_gguf_bytes(version, &kvs, &tensors, 0);
            let file_mem = GgufFile::from_bytes(bytes.clone()).expect("parse from bytes");
            assert_eq!(file_mem.version(), version);
            assert_eq!(file_mem.alignment(), 64);
            assert_eq!(file_mem.tensor_count(), 3);
            assert_eq!(file_mem.metadata_kv_count(), kvs.len() as u64);
            file_mem.check_brackets().expect("brackets valid");

            let path = dir.path().join(format!("test-v{version}.gguf"));
            fs::write(&path, &bytes).expect("write gguf");
            let file_mmap = GgufFile::open(&path).expect("open via mmap");
            assert_eq!(file_mmap.version(), version);
            assert_eq!(file_mmap.get_str("general.architecture"), Some("qwen4exp"));
            assert_eq!(file_mmap.get_u64("test.u8"), Some(250));
            assert_eq!(file_mmap.get_i64("test.i8"), Some(-42));
            assert_eq!(file_mmap.get_u64("test.u16"), Some(65000));
            assert_eq!(file_mmap.get_i64("test.i16"), Some(-12345));
            assert_eq!(file_mmap.get_u64("test.u32"), Some(12345678));
            assert_eq!(file_mmap.get_i64("test.i32"), Some(-87654321));
            assert_eq!(file_mmap.get_f64("test.f32"), Some(3.5));
            assert_eq!(file_mmap.get_bool("test.bool_true"), Some(true));
            assert_eq!(file_mmap.get_bool("test.bool_false"), Some(false));
            assert_eq!(file_mmap.get_u64("test.u64"), Some(1 << 40));
            assert_eq!(file_mmap.get_i64("test.i64"), Some(-(1i64 << 40)));
            assert_eq!(file_mmap.get_f64("test.f64"), Some(1.23456789));

            let t0 = file_mmap
                .find_tensor("token_embd.weight")
                .expect("find token_embd");
            assert_eq!(t0.n_dims, 2);
            assert_eq!(t0.dims, [64, 16, 1, 1]);
            assert_eq!(t0.shape(), &[64, 16]);
            assert_eq!(t0.qtype, QuantType::Q8_0);
            assert_eq!(t0.byte_size, (64 * 16 / 32) * 34);
            let slice = file_mmap
                .tensor_bytes("token_embd.weight")
                .expect("tensor slice");
            assert_eq!(slice.len(), t0.byte_size as usize);
            for (idx, &b) in slice.iter().enumerate() {
                assert_eq!(b, pattern_byte(1, idx));
            }

            let view: GgufView = (&file_mmap).into();
            assert_eq!(view.architecture, "qwen4exp");
            assert_eq!(view.tensor_count, 3);
            assert_eq!(view.metadata_u64.get("test.u32"), Some(&12345678));
            assert_eq!(view.metadata_f64.get("test.f32"), Some(&3.5));
        }
    }

    #[test]
    fn gguf_error_cases_rejected() {
        // 1. Bad magic
        let mut bad_magic = build_gguf_bytes(3, &[], &[], 0);
        bad_magic[0] = b'X';
        assert!(matches!(
            GgufFile::from_bytes(bad_magic),
            Err(GgufError::InvalidMagic { .. })
        ));

        // 2. Unsupported version (v1 and v4)
        for bad_ver in [1u32, 4u32] {
            let b = build_gguf_bytes(bad_ver, &[], &[], 0);
            assert!(matches!(
                GgufFile::from_bytes(b),
                Err(GgufError::UnsupportedVersion {
                    version,
                    ..
                }) if version == bad_ver
            ));
        }

        // 3. Duplicate tensor name in single file
        let dup_tensors = vec![
            SyntheticTensor {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![32],
                qtype: QuantType::F32,
                seed: 1,
            },
            SyntheticTensor {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![32],
                qtype: QuantType::F32,
                seed: 2,
            },
        ];
        let dup_bytes = build_gguf_bytes(3, &[], &dup_tensors, 0);
        assert!(matches!(
            GgufFile::from_bytes(dup_bytes),
            Err(GgufError::DuplicateTensorName { tensor, .. }) if tensor == "blk.0.attn_q.weight"
        ));

        // 4. Duplicate metadata key
        let dup_kv = build_gguf_bytes(
            3,
            &[
                ("general.name", GgufValue::U32(1)),
                ("general.name", GgufValue::U32(2)),
            ],
            &[],
            0,
        );
        assert!(matches!(
            GgufFile::from_bytes(dup_kv),
            Err(GgufError::DuplicateMetadataKey { key, .. }) if key == "general.name"
        ));

        // 5. Invalid alignment (not power of two)
        let mut bad_align = Vec::new();
        bad_align.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bad_align.extend_from_slice(&3u32.to_le_bytes());
        bad_align.extend_from_slice(&0u64.to_le_bytes());
        bad_align.extend_from_slice(&1u64.to_le_bytes());
        push_str(&mut bad_align, "general.alignment");
        push_value(&mut bad_align, &GgufValue::U32(48));
        assert!(matches!(
            GgufFile::from_bytes(bad_align),
            Err(GgufError::InvalidAlignment { alignment: 48, .. })
        ));

        // 6. Invalid UTF-8 in metadata key
        let mut bad_utf8 = Vec::new();
        bad_utf8.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bad_utf8.extend_from_slice(&3u32.to_le_bytes());
        bad_utf8.extend_from_slice(&0u64.to_le_bytes());
        bad_utf8.extend_from_slice(&1u64.to_le_bytes());
        bad_utf8.extend_from_slice(&4u64.to_le_bytes());
        bad_utf8.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
        assert!(matches!(
            GgufFile::from_bytes(bad_utf8),
            Err(GgufError::InvalidUtf8 { .. })
        ));

        // 7. Invalid metadata value type tag
        let mut bad_vtype = Vec::new();
        bad_vtype.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bad_vtype.extend_from_slice(&3u32.to_le_bytes());
        bad_vtype.extend_from_slice(&0u64.to_le_bytes());
        bad_vtype.extend_from_slice(&1u64.to_le_bytes());
        push_str(&mut bad_vtype, "bad.key");
        bad_vtype.extend_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            GgufFile::from_bytes(bad_vtype),
            Err(GgufError::InvalidValueType { type_id: 99, .. })
        ));

        // 8. Truncated tensor payload
        let valid_tensors = vec![SyntheticTensor {
            name: "blk.0.w".to_string(),
            dims: vec![64, 4],
            qtype: QuantType::Q8_0,
            seed: 5,
        }];
        let truncated_data = build_gguf_bytes(3, &[], &valid_tensors, 10);
        assert!(matches!(
            GgufFile::from_bytes(truncated_data),
            Err(GgufError::TruncatedData { tensor, .. }) if tensor == "blk.0.w"
        ));

        // 9. Truncated header
        let full = build_gguf_bytes(3, &[("a.b", GgufValue::U64(1))], &valid_tensors, 0);
        assert!(matches!(
            GgufFile::from_bytes(&full[..12]),
            Err(GgufError::TruncatedHeader { .. })
        ));
    }

    // ------------------------------------------------------------------------
    // 2. Multi-shard GgufModel tests (gguf_split_test.cpp parity)
    // ------------------------------------------------------------------------

    fn write_three_shard_fixture<F>(dir: &Path, mut mutate: F) -> Vec<PathBuf>
    where
        F: FnMut(usize, &mut Vec<(&'static str, GgufValue)>, &mut Vec<SyntheticTensor>, &mut usize),
    {
        let mut kvs: [Vec<(&'static str, GgufValue)>; 3] = [
            vec![
                (
                    "general.architecture",
                    GgufValue::String("qwen4exp".to_string()),
                ),
                ("qwen4exp.block_count", GgufValue::U32(48)),
                ("qwen4exp.embedding_length", GgufValue::U32(2560)),
                ("split.no", GgufValue::U16(0)),
                ("split.count", GgufValue::U16(3)),
                ("split.tensors.count", GgufValue::I32(4)),
            ],
            vec![
                ("split.no", GgufValue::U16(1)),
                ("split.count", GgufValue::U16(3)),
                ("split.tensors.count", GgufValue::I32(4)),
            ],
            vec![
                ("split.no", GgufValue::U16(2)),
                ("split.count", GgufValue::U16(3)),
                ("split.tensors.count", GgufValue::I32(4)),
            ],
        ];

        // Shard 1 is metadata-only (0 tensors), Shard 2 has 2 tensors, Shard 3 has 2 tensors.
        let mut ts: [Vec<SyntheticTensor>; 3] = [
            vec![],
            vec![
                SyntheticTensor {
                    name: "token_embd.weight".to_string(),
                    dims: vec![64, 8],
                    qtype: QuantType::F32,
                    seed: 1,
                },
                SyntheticTensor {
                    name: "blk.0.attn_qkv.weight".to_string(),
                    dims: vec![64, 12],
                    qtype: QuantType::Q8_0,
                    seed: 2,
                },
            ],
            vec![
                SyntheticTensor {
                    name: "blk.11.ffn_gate_exps.weight".to_string(),
                    dims: vec![256, 4, 2],
                    qtype: QuantType::Q5_K,
                    seed: 3,
                },
                SyntheticTensor {
                    name: "blk.12.attn_q.weight".to_string(),
                    dims: vec![256, 4],
                    qtype: QuantType::Q6_K,
                    seed: 4,
                },
            ],
        ];

        let mut out = Vec::with_capacity(3);
        for i in 0..3 {
            let mut cut = 0usize;
            mutate(i, &mut kvs[i], &mut ts[i], &mut cut);
            let p = dir.join(format!("Model-UD-Q5_K_XL-{:05}-of-00003.gguf", i + 1));
            let bytes = build_gguf_bytes(3, &kvs[i], &ts[i], cut);
            fs::write(&p, bytes).expect("write shard");
            out.push(p);
        }
        out
    }

    #[test]
    fn split_shards_discovery_combining_and_rejection() {
        // 1. Valid 3-shard family discovered from shard 1 or shard 3
        {
            let dir = ScratchDir::new("split-ok");
            let paths = write_three_shard_fixture(dir.path(), |_, _, _, _| {});
            let from_s1 = gguf_split_paths(&paths[0]).expect("discover from shard 1");
            let from_s3 = gguf_split_paths(&paths[2]).expect("discover from shard 3");
            assert_eq!(from_s1, paths);
            assert_eq!(from_s3, paths);

            let model = GgufModel::open(&paths[2]).expect("open from shard 3");
            assert_eq!(model.shard_count(), 3);
            assert_eq!(model.meta_shard().tensor_count(), 0);
            assert_eq!(
                model.meta_shard().data_start(),
                model.meta_shard().file_size()
            );
            assert_eq!(model.tensor_count(), 4);

            let loc_qkv = model
                .tensor_location("blk.0.attn_qkv.weight")
                .expect("loc qkv");
            assert_eq!(loc_qkv.shard_idx, 1);

            let loc_q = model
                .tensor_location("blk.12.attn_q.weight")
                .expect("loc q");
            assert_eq!(loc_q.shard_idx, 2);

            for (name, seed) in [
                ("token_embd.weight", 1u8),
                ("blk.0.attn_qkv.weight", 2u8),
                ("blk.11.ffn_gate_exps.weight", 3u8),
                ("blk.12.attn_q.weight", 4u8),
            ] {
                let data = model.tensor_bytes(name).expect("tensor_bytes");
                for (idx, &b) in data.iter().enumerate() {
                    assert_eq!(b, pattern_byte(seed, idx));
                }
            }

            let view = model.to_gguf_view();
            assert_eq!(view.architecture, "qwen4exp");
            assert_eq!(view.tensor_count, 4);
            assert_eq!(view.metadata_u64.get("qwen4exp.block_count"), Some(&48));
        }

        // 2. Missing shard 2 of 3 is rejected naming the missing shard
        {
            let dir = ScratchDir::new("split-missing");
            let paths = write_three_shard_fixture(dir.path(), |_, _, _, _| {});
            fs::remove_file(&paths[1]).unwrap();
            let err = GgufModel::open(&paths[0]).unwrap_err();
            assert!(matches!(
                err,
                GgufError::MissingShard {
                    shard_no: 2,
                    shard_count: 3,
                    ..
                }
            ));
        }

        // 3. Single shard with split.count > 1 opened without sibling shards is rejected
        {
            let dir = ScratchDir::new("split-lone");
            let paths = write_three_shard_fixture(dir.path(), |_, _, _, _| {});
            let lone = dir.path().join("renamed-single.gguf");
            fs::rename(&paths[0], &lone).unwrap();
            let err = GgufModel::open(&lone).unwrap_err();
            assert!(err.to_string().contains("shard 1 of 3"));
        }

        // 4. Swapped shards (wrong split.no) rejected
        {
            let dir = ScratchDir::new("split-swapped");
            let paths = write_three_shard_fixture(dir.path(), |_, _, _, _| {});
            let err = GgufModel::open_paths([&paths[0], &paths[2], &paths[1]]).unwrap_err();
            assert!(matches!(err, GgufError::InvalidSplit { .. }));
        }

        // 5. Wrong split.count in shard 3 rejected
        {
            let dir = ScratchDir::new("split-bad-count");
            let paths = write_three_shard_fixture(dir.path(), |i, kv, _, _| {
                if i == 2 {
                    for (k, v) in kv.iter_mut() {
                        if *k == "split.count" {
                            *v = GgufValue::U16(4);
                        }
                    }
                }
            });
            assert!(matches!(
                GgufModel::open_paths(&paths),
                Err(GgufError::InvalidSplit { .. })
            ));
        }

        // 6. Wrong split.tensors.count rejected
        {
            let dir = ScratchDir::new("split-bad-tcount");
            let paths = write_three_shard_fixture(dir.path(), |_, kv, _, _| {
                for (k, v) in kv.iter_mut() {
                    if *k == "split.tensors.count" {
                        *v = GgufValue::I32(99);
                    }
                }
            });
            let err = GgufModel::open_paths(&paths).unwrap_err();
            assert!(err.to_string().contains("99"));
        }

        // 7. Missing general.architecture in shard 1 rejected
        {
            let dir = ScratchDir::new("split-no-arch");
            let paths = write_three_shard_fixture(dir.path(), |i, kv, _, _| {
                if i == 0 {
                    kv.retain(|(k, _)| *k != "general.architecture");
                }
            });
            let err = GgufModel::open_paths(&paths).unwrap_err();
            assert!(err.to_string().contains("general.architecture"));
        }

        // 8. Duplicate tensor across shard 2 and shard 3 rejected
        {
            let dir = ScratchDir::new("split-dup-tensor");
            let paths = write_three_shard_fixture(dir.path(), |i, _, ts, _| {
                if i == 2 {
                    ts.push(SyntheticTensor {
                        name: "token_embd.weight".to_string(),
                        dims: vec![64, 8],
                        qtype: QuantType::F32,
                        seed: 9,
                    });
                }
                if i == 1 {
                    ts.pop();
                }
            });
            let err = GgufModel::open_paths(&paths).unwrap_err();
            assert!(matches!(
                err,
                GgufError::DuplicateTensorName { tensor, .. } if tensor == "token_embd.weight"
            ));
        }

        // 9. Conflicting metadata key across shards rejected
        {
            let dir = ScratchDir::new("split-conflict-meta");
            let paths = write_three_shard_fixture(dir.path(), |i, kv, _, _| {
                if i == 1 {
                    kv.push(("qwen4exp.block_count", GgufValue::U32(99)));
                }
            });
            let err = GgufModel::open_paths(&paths).unwrap_err();
            assert!(matches!(
                err,
                GgufError::ConflictingMetadata { key, .. } if key == "qwen4exp.block_count"
            ));
        }

        // 10. Truncated tensor in shard 3 rejected
        {
            let dir = ScratchDir::new("split-truncated");
            let paths = write_three_shard_fixture(dir.path(), |i, _, _, cut| {
                if i == 2 {
                    *cut = 100;
                }
            });
            let err = GgufModel::open_paths(&paths).unwrap_err();
            assert!(matches!(
                err,
                GgufError::TruncatedData { tensor, .. } if tensor == "blk.12.attn_q.weight"
            ));
        }
    }

    // ------------------------------------------------------------------------
    // 3. Strata native_experts.txt (v2-v4 + tests/data/native_experts fixtures)
    //    and PackIndex (index.txt + dense.bin)
    // ------------------------------------------------------------------------

    #[test]
    fn repo_native_experts_fixtures_parse_and_validate() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let data_dir = manifest_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests")
            .join("data")
            .join("native_experts");

        struct Case {
            file: &'static str,
            n_expert: usize,
            total_bytes: u64,
            l0_gu: i32,
            l0_d: i32,
            l0_bytes: usize,
            l3_gu: i32,
            l3_d: i32,
            l3_bytes: usize,
        }

        let cases = [
            Case {
                file: "q2_0.txt",
                n_expert: 512,
                total_bytes: 33_973_862_400,
                l0_gu: 42,
                l0_d: 42,
                l0_bytes: 1_382_400,
                l3_gu: 42,
                l3_d: 42,
                l3_bytes: 1_382_400,
            },
            Case {
                file: "iq2_xs.txt",
                n_expert: 512,
                total_bytes: 35_454_976_000,
                l0_gu: 22,
                l0_d: 42,
                l0_bytes: 1_510_400,
                l3_gu: 22,
                l3_d: 42,
                l3_bytes: 1_510_400,
            },
            Case {
                file: "iq3_xxs.txt",
                n_expert: 512,
                total_bytes: 42_912_972_800,
                l0_gu: 17,
                l0_d: 20,
                l0_bytes: 1_868_800,
                l3_gu: 16,
                l3_d: 42,
                l3_bytes: 1_305_600,
            },
            Case {
                file: "iq3_s.txt",
                n_expert: 512,
                total_bytes: 50_292_326_400,
                l0_gu: 18,
                l0_d: 20,
                l0_bytes: 2_176_000,
                l3_gu: 22,
                l3_d: 20,
                l3_bytes: 1_971_200,
            },
            Case {
                file: "coder-iq1_m.txt",
                n_expert: 256,
                total_bytes: 25_146_163_200,
                l0_gu: 18,
                l0_d: 20,
                l0_bytes: 2_176_000,
                l3_gu: 22,
                l3_d: 20,
                l3_bytes: 1_971_200,
            },
            Case {
                file: "swift-iq2_xs.txt",
                n_expert: 512,
                total_bytes: 35_454_976_000,
                l0_gu: 22,
                l0_d: 42,
                l0_bytes: 1_510_400,
                l3_gu: 22,
                l3_d: 42,
                l3_bytes: 1_510_400,
            },
        ];

        for c in cases {
            let layout = ExpertLayout::open(data_dir.join(c.file)).expect(c.file);
            assert_eq!(layout.n_layers, 48);
            assert_eq!(layout.n_expert, c.n_expert);
            assert_eq!(layout.total_bytes, c.total_bytes);
            assert_eq!(layout.layer(0).unwrap().fmt.gu_type, c.l0_gu);
            assert_eq!(layout.layer(0).unwrap().fmt.d_type, c.l0_d);
            assert_eq!(layout.layer(0).unwrap().fmt.bytes, c.l0_bytes);
            assert_eq!(layout.layer(3).unwrap().fmt.gu_type, c.l3_gu);
            assert_eq!(layout.layer(3).unwrap().fmt.d_type, c.l3_d);
            assert_eq!(layout.layer(3).unwrap().fmt.bytes, c.l3_bytes);
        }
    }

    #[test]
    fn native_experts_v2_v3_v4_and_gguf_assembly() {
        // Small dimensions: H=256, FF=256, n_layers=3, n_expert=4
        let cfg = ExpertLayoutConfig {
            n_layers: 3,
            n_expert: 4,
            n_embd: 256,
            n_ff: 256,
        };
        let f_q5k = NativeFmt::compute(13, 13, 256, 256).unwrap();
        let f_q6k = NativeFmt::compute(14, 14, 256, 256).unwrap();
        let mut txt = String::from(
            "# strata native experts v4: gate[FF,H] || up[FF,H] || down[H,FF] per (layer, expert) (n_expert 4, total 0)\n",
        );
        let s1 = "Model-00001-of-00003.gguf";
        let s2 = "Model-00002-of-00003.gguf";
        let s3 = "Model-00003-of-00003.gguf";

        let dir = ScratchDir::new("expert-v4");
        let kvs0 = vec![
            (
                "general.architecture",
                GgufValue::String("qwen4exp".to_string()),
            ),
            ("split.no", GgufValue::U16(0)),
            ("split.count", GgufValue::U16(3)),
            ("split.tensors.count", GgufValue::I32(9)),
        ];
        let kvs1 = vec![
            ("split.no", GgufValue::U16(1)),
            ("split.count", GgufValue::U16(3)),
            ("split.tensors.count", GgufValue::I32(9)),
        ];
        let kvs2 = vec![
            ("split.no", GgufValue::U16(2)),
            ("split.count", GgufValue::U16(3)),
            ("split.tensors.count", GgufValue::I32(9)),
        ];

        // Layer 0 in shard 1, Layer 1 split across shard 1 (gate, up) and shard 2 (down), Layer 2 in shard 3
        let ts0 = vec![
            SyntheticTensor {
                name: "blk.0.ffn_gate_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 10,
            },
            SyntheticTensor {
                name: "blk.0.ffn_up_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 11,
            },
            SyntheticTensor {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 12,
            },
            SyntheticTensor {
                name: "blk.1.ffn_gate_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q6_K,
                seed: 20,
            },
            SyntheticTensor {
                name: "blk.1.ffn_up_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q6_K,
                seed: 21,
            },
        ];
        let ts1 = vec![SyntheticTensor {
            name: "blk.1.ffn_down_exps.weight".to_string(),
            dims: vec![256, 256, 4],
            qtype: QuantType::Q6_K,
            seed: 22,
        }];
        let ts2 = vec![
            SyntheticTensor {
                name: "blk.2.ffn_gate_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 30,
            },
            SyntheticTensor {
                name: "blk.2.ffn_up_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 31,
            },
            SyntheticTensor {
                name: "blk.2.ffn_down_exps.weight".to_string(),
                dims: vec![256, 256, 4],
                qtype: QuantType::Q5_K,
                seed: 32,
            },
        ];

        let p0 = dir.path().join(s1);
        let p1 = dir.path().join(s2);
        let p2 = dir.path().join(s3);
        fs::write(&p0, build_gguf_bytes(3, &kvs0, &ts0, 0)).unwrap();
        fs::write(&p1, build_gguf_bytes(3, &kvs1, &ts1, 0)).unwrap();
        fs::write(&p2, build_gguf_bytes(3, &kvs2, &ts2, 0)).unwrap();

        let model = GgufModel::open(&p0).unwrap();
        let file_off = |name: &str| -> u64 {
            let (e, loc) = model.find_tensor(name).unwrap();
            model.shards()[loc.shard_idx].data_start() + e.offset
        };

        let mut running = 0u64;
        // Layer 0: v2/v3 empty shard col (uses primary shard 1)
        txt.push_str(&format!(
            "0 13 13 {running} {} {} {} {}\n",
            f_q5k.bytes,
            file_off("blk.0.ffn_gate_exps.weight"),
            file_off("blk.0.ffn_up_exps.weight"),
            file_off("blk.0.ffn_down_exps.weight")
        ));
        running += (f_q5k.bytes * 4) as u64;
        // Layer 1: v4 comma-separated shard column (gate & up in shard 1 = empty, down in shard 2)
        txt.push_str(&format!(
            "1 14 14 {running} {} {} {} {} ,,{s2}\n",
            f_q6k.bytes,
            file_off("blk.1.ffn_gate_exps.weight"),
            file_off("blk.1.ffn_up_exps.weight"),
            file_off("blk.1.ffn_down_exps.weight")
        ));
        running += (f_q6k.bytes * 4) as u64;
        // Layer 2: v3 single shard column (all 3 in shard 3)
        txt.push_str(&format!(
            "2 13 13 {running} {} {} {} {} {s3}\n",
            f_q5k.bytes,
            file_off("blk.2.ffn_gate_exps.weight"),
            file_off("blk.2.ffn_up_exps.weight"),
            file_off("blk.2.ffn_down_exps.weight")
        ));

        let layout = ExpertLayout::parse_str_with_config(&txt, cfg).expect("parse v4 layout");
        assert_eq!(layout.version, 4);
        assert_eq!(layout.n_expert, 4);
        assert!(layout.has_gguf_offsets);
        layout
            .check_gguf(model.shards(), Some(&p0))
            .expect("check_gguf passes");

        // Assemble layer 1 expert 2 (split across shard 1 and shard 2) and verify every byte!
        let blob = layout
            .assemble_expert_from_gguf(model.shards(), Some(&p0), 1, 2)
            .expect("assemble expert (1, 2)");
        assert_eq!(blob.len(), f_q6k.bytes);
        let role_len = f_q6k.up_off;
        for b in 0..role_len {
            assert_eq!(blob[b], pattern_byte(20, 2 * role_len + b));
            assert_eq!(blob[role_len + b], pattern_byte(21, 2 * role_len + b));
            assert_eq!(blob[2 * role_len + b], pattern_byte(22, 2 * role_len + b));
        }

        // Reject v5 and malformed shard columns
        let v5_txt = txt.replacen("v4:", "v5:", 1);
        assert!(ExpertLayout::parse_str_with_config(&v5_txt, cfg).is_err());
        let bad_comma = txt.replacen(&format!(",,{s2}"), &format!(",{s2}"), 1);
        assert!(ExpertLayout::parse_str_with_config(&bad_comma, cfg).is_err());
    }

    #[test]
    fn strata_pack_index_and_dense_bin_reader() {
        let dir = ScratchDir::new("strata-pack");
        let idx_txt = "\
# strata pack index v3
# align 256 pool 2304 tensors 2
blk.0.attn_qkv.weight 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0
blk.1.ple_key.weight 0 4 0 2048 256 2048 256 4 0 0 1 0 0 0 0 0 0 0
";
        fs::write(dir.path().join("index.txt"), idx_txt).unwrap();
        let dense_bytes: Vec<u8> = (0..2048).map(|i| pattern_byte(9, i)).collect();
        fs::write(dir.path().join("dense.bin"), &dense_bytes).unwrap();

        let pack = StrataPack::open(dir.path()).expect("open StrataPack");
        let idx = pack.index().expect("has index");
        assert_eq!(idx.align, 256);
        assert_eq!(idx.pool_bytes_total, 2304);
        assert_eq!(idx.index_code_bits("blk.0.attn_qkv.weight"), Some(8));
        assert_eq!(idx.index_code_bits("blk.1.ple_key.weight"), Some(0));
        assert_eq!(idx.index_code_bits("missing.weight"), None);

        let mut skip = BTreeSet::new();
        skip.insert("blk.0.attn_qkv.weight".to_string());
        assert_eq!(idx.pool_bytes(Some(&skip)), 2048);

        let key_slice = pack
            .dense_tensor_bytes("blk.1.ple_key.weight")
            .expect("dense_tensor_bytes");
        assert_eq!(key_slice, dense_bytes.as_slice());
    }

    // ------------------------------------------------------------------------
    // 4. DraftVocabFile and MtpRtPack tests
    // ------------------------------------------------------------------------

    #[test]
    fn draft_vocab_round_trip_and_validation() {
        let dir = ScratchDir::new("draft-vocab");
        let bin_path = dir.path().join("draft_vocab.bin");

        // 65,536-entry vocabulary round-trip
        let ids: Vec<u32> = (0..65_536u32).map(|i| i * 3 + 1).collect();
        let vocab = DraftVocab::new(ids.clone());
        DraftVocabFile::write(&bin_path, &vocab).expect("write draft_vocab.bin");

        let loaded = DraftVocabFile::open_checked(&bin_path, Some(248_320)).expect("open_checked");
        assert_eq!(loaded.len(), 65_536);
        assert_eq!(loaded.draft_to_target, ids);

        // Out-of-range target vocab check
        assert!(DraftVocabFile::open_checked(&bin_path, Some(100_000)).is_err());

        // Non-multiple of 4 bytes rejected
        assert!(DraftVocabFile::from_bytes(&[1, 2, 3, 4, 5]).is_err());

        // Negative token ID rejected
        assert!(DraftVocabFile::from_bytes(&(-1i32).to_le_bytes()).is_err());
    }

    #[test]
    fn mtp_rt_pack_reader() {
        let dir = ScratchDir::new("mtp-rt");
        // Build a synthetic dense.txt + dense.bin + draft_vocab.bin
        // q8_0 row of 1x64 = 2 blocks = 68 bytes; bf16 2x16 = 64 bytes; f32 1x16 = 64 bytes
        let dense_txt = "\
fc_embedding.weight q8_0 1 64 0 68
enorm.weight bf16 2 16 128 64
mlp.gate.weight f32 1 16 192 64
";
        fs::write(dir.path().join("dense.txt"), dense_txt).unwrap();
        let mut dense_bin = vec![0u8; 256];
        for (i, dst) in dense_bin.iter_mut().enumerate().take(68) {
            *dst = pattern_byte(1, i);
        }
        for i in 0..64 {
            dense_bin[128 + i] = pattern_byte(2, i);
        }
        for i in 0..64 {
            dense_bin[192 + i] = pattern_byte(3, i);
        }
        fs::write(dir.path().join("dense.bin"), &dense_bin).unwrap();
        DraftVocabFile::write(
            dir.path().join("draft_vocab.bin"),
            &DraftVocab::new(vec![10, 20, 30, 40]),
        )
        .unwrap();

        let mtp = MtpRtPack::open(dir.path()).expect("open MtpRtPack");
        assert_eq!(mtp.tensors().len(), 3);
        assert_eq!(
            mtp.tensor_bytes("fc_embedding.weight").unwrap(),
            &dense_bin[0..68]
        );
        assert_eq!(
            mtp.tensor_bytes("enorm.weight").unwrap(),
            &dense_bin[128..192]
        );
        assert_eq!(
            mtp.tensor_bytes("mlp.gate.weight").unwrap(),
            &dense_bin[192..256]
        );
        assert_eq!(
            mtp.draft_vocab().unwrap().draft_to_target,
            vec![10, 20, 30, 40]
        );
        // Incomplete tensor set should fail `validate_required_tensors`
        assert!(mtp.validate_required_tensors().is_err());
    }

    // ------------------------------------------------------------------------
    // 5. Real GGUF headers and 5 model JSON shape catalogs in TALUS_SHAPES_DIR
    // ------------------------------------------------------------------------

    #[test]
    fn real_matrix_shapes_and_gguf_headers_validate() {
        let shapes_dir = std::env::var("TALUS_SHAPES_DIR").ok().map(PathBuf::from);
        let Some(shapes_dir) = shapes_dir.filter(|d| d.is_dir()) else {
            eprintln!("skipping real_matrix_shapes_and_gguf_headers_validate: shape catalog directory not present");
            return;
        };

        let summaries =
            validate_matrix_shapes_dir(&shapes_dir).expect("all 5 models + headers must validate");
        assert_eq!(summaries.len(), 5);

        let total_tensors: usize = summaries.iter().map(|s| s.tensors_checked).sum();
        let total_shards: usize = summaries.iter().map(|s| s.shards_checked).sum();
        assert_eq!(total_tensors, 5064);
        assert_eq!(total_shards, 18);

        // Also verify standalone MTP header in hdr/mtp-Qwen3.8-27B-Q4_0.gguf.hdr if present
        let mtp_q4_hdr = shapes_dir.join("hdr/mtp-Qwen3.8-27B-Q4_0.gguf.hdr");
        if mtp_q4_hdr.is_file() {
            let f = GgufFile::open(&mtp_q4_hdr).expect("open mtp-Qwen3.8-27B-Q4_0.gguf.hdr");
            f.check_brackets().expect("mtp brackets valid");
            assert_eq!(f.tensor_count(), 18);
        }
    }
}
