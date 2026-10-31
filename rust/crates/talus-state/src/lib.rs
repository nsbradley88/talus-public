//! Sequence state, recurrent snapshots, KV stash, prefix cache, and disk sessions.
//!
//! Implements WP-2c:
//! - [`EngineStateView`], [`PromptCache`], [`Snap`], [`Stash`], [`RestoreOutcome`]
//! - Byte-size and asynchronous HIP copy helpers: [`vt_round`], [`state_bytes`], [`state_copy`],
//!   [`kv_bytes`], [`kv_copy`], [`hip_memcpy_2d_async`]
//! - Message-boundary chunking helpers: [`find_message_boundary`], [`find_msg_cut`]
//! - [`talus_core_api::PrefixCache`] implementation for [`PromptCache`] and [`TalusPrefixCache`]
//! - Per-layer [`Q36LayerStateView`] and [`Q36PromptCache`]
//! - Persistent disk tier ([`disk::DiskCache`]) and its Qwen3.5/3.6 adapter ([`q36_disk`])

pub mod disk;
pub mod pcache;
pub mod prefix_cache;
pub mod q36_cache;
pub mod q36_disk;

pub use pcache::{
    find_message_boundary, find_msg_cut, hip_memcpy_2d_async, kv_bytes, kv_copy, state_bytes,
    state_copy, vt_round, EngineStateTarget, EngineStateView, PromptCache, RestoreOutcome, Snap,
    Stash, DEFAULT_MAX_MSG, DEFAULT_MAX_SNAPS, DEFAULT_SNAP_MSG_MIN, DEFAULT_STASH_BUDGET,
    IM_START_TOKEN_ID, TOOL_ROLE_TOKEN_ID,
};
pub use prefix_cache::{
    compute_i32_tokens_hash, compute_tokens_hash, PrefixMetaEntry, TalusPrefixCache,
};
pub use q36_cache::{
    q36_kv_bytes, q36_kv_copy, q36_kv_copy_range, q36_state_bytes, q36_state_copy, Q36KvCacheSlice,
    Q36LayerStateView, Q36PromptCache, Q36Snap, Q36Stash, Q36StateSnapshot,
    DEFAULT_Q36_STASH_BUDGET,
};

pub const CRATE_NAME: &str = "talus-state";

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::c_void;
    use std::ptr;

    use hip_sys::{
        hipFree, hipMalloc, hipMemcpy, hipMemcpyDeviceToHost, hipMemcpyHostToDevice, hipMemset,
        hipStreamCreateWithFlags, hipStreamDestroy, hipStreamNonBlocking, hipStream_t,
    };
    use talus_core_api::{PrefixCache, SnapReason, Snapshot};

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-state");
    }

    #[test]
    fn test_vt_round_and_byte_helpers() {
        assert_eq!(vt_round(0), 0);
        assert_eq!(vt_round(1), 16);
        assert_eq!(vt_round(15), 16);
        assert_eq!(vt_round(16), 16);
        assert_eq!(vt_round(17), 32);
        assert_eq!(vt_round(31), 32);
        assert_eq!(vt_round(32), 32);
        assert_eq!(vt_round(33), 48);

        let mut eng = EngineStateView {
            n_gdn: 3,
            n_attn: 2,
            gdn_h: 4,
            gdn_d: 8,
            gdn_s: 8,
            gdn_q: 32,
            gdn_c: 4,
            n_kv: 2,
            d_head: 16,
            max_seq: 64,
            d_model: 64,
            use_bf16_kv: true,
            ..Default::default()
        };

        // gdn_bytes = 3 * 4 * 8 * 8 * 4 = 3072
        // conv_bytes = 3 * (4 - 1) * 32 * 4 = 1152
        // mtp_bytes = 64 * 4 = 256
        // x_bytes = 64 * 4 = 256
        assert_eq!(eng.gdn_bytes(), 3072);
        assert_eq!(eng.conv_bytes(), 1152);
        assert_eq!(eng.mtp_bytes(), 256);
        assert_eq!(eng.x_bytes(), 256);
        assert_eq!(state_bytes(&eng), 3072 + 1152 + 256 + 256);

        // row_bytes = 2 * 16 * 2 = 64
        // l = 20 -> k_bytes = 2 * 20 * 64 = 2560
        // use_bf16_kv = true -> v_bytes = 2560 -> total = 5120
        assert_eq!(kv_bytes(&eng, 20), 5120);

        // use_bf16_kv = false -> vt_round(20) = 32
        // v_bytes = 2 * 16 * 2 * 32 * 2 = 4096 -> total = 2560 + 4096 = 6656
        eng.use_bf16_kv = false;
        assert_eq!(kv_bytes(&eng, 20), 2560 + 4096);
        assert_eq!(kv_bytes(&eng, 0), 0);
    }

    #[test]
    fn test_message_boundary_finder() {
        let mut prompt = vec![10i32; 1200];
        // Too close to reuse (distance 500 < 1024):
        prompt[500] = IM_START_TOKEN_ID;
        prompt[501] = 846; // user role
        assert_eq!(
            find_message_boundary(&prompt, 0, DEFAULT_SNAP_MSG_MIN),
            None
        );

        // At index 1050, `<|im_start|>` followed by `tool` (74455) must be ignored:
        prompt[1050] = IM_START_TOKEN_ID;
        prompt[1051] = TOOL_ROLE_TOKEN_ID;
        assert_eq!(
            find_message_boundary(&prompt, 0, DEFAULT_SNAP_MSG_MIN),
            None
        );

        // At index 1100, `<|im_start|>` followed by non-tool (e.g., user 846) should be selected:
        prompt[1100] = IM_START_TOKEN_ID;
        prompt[1101] = 846;
        assert_eq!(
            find_message_boundary(&prompt, 0, DEFAULT_SNAP_MSG_MIN),
            Some(1100)
        );

        // Later valid `<|im_start|>` at 1150 should supersede 1100, even if followed by tool at 1180:
        prompt[1150] = IM_START_TOKEN_ID;
        prompt[1151] = 846;
        prompt[1180] = IM_START_TOKEN_ID;
        prompt[1181] = TOOL_ROLE_TOKEN_ID;
        assert_eq!(
            find_message_boundary(&prompt, 0, DEFAULT_SNAP_MSG_MIN),
            Some(1150)
        );

        // If `reuse` advances to 200, distance 1150 - 200 = 950 < 1024 -> None:
        assert_eq!(
            find_message_boundary(&prompt, 200, DEFAULT_SNAP_MSG_MIN),
            None
        );
        // With threshold 500, 1150 - 200 = 950 >= 500 -> Some(1150):
        assert_eq!(find_message_boundary(&prompt, 200, 500), Some(1150));
    }

    /// Small GPU mock wrapper owning device allocations (`< 64 KiB` total, well within the 64 MiB limit).
    struct MockGpuEngine {
        view: EngineStateView,
        gdn_len: usize,
        conv_len: usize,
        mtp_len: usize,
        x_len: usize,
        k_u16_len: usize,
        v_u16_len: usize,
    }

    impl MockGpuEngine {
        fn new(use_bf16_kv: bool) -> Self {
            let n_gdn = 2usize;
            let n_attn = 2usize;
            let gdn_h = 2i32;
            let gdn_d = 4i32;
            let gdn_s = 4i32;
            let gdn_q = 8i32;
            let gdn_c = 4i32;
            let n_kv = 2i32;
            let d_head = 8i32;
            let max_seq = 32i32;
            let d_model = 16i32;

            let gdn_len = n_gdn * (gdn_h as usize) * (gdn_d as usize) * (gdn_s as usize);
            let conv_len = n_gdn * ((gdn_c - 1) as usize) * (gdn_q as usize);
            let mtp_len = d_model as usize;
            let x_len = d_model as usize;
            let k_u16_len = n_attn * (max_seq as usize) * (n_kv as usize) * (d_head as usize);
            let v_u16_len = k_u16_len;

            unsafe {
                let mut s_main: hipStream_t = ptr::null_mut();
                assert!(hipStreamCreateWithFlags(&mut s_main, hipStreamNonBlocking).is_success());

                let alloc_f32 = |count: usize| -> *mut f32 {
                    let mut p: *mut c_void = ptr::null_mut();
                    assert!(hipMalloc(&mut p, count * 4).is_success());
                    assert!(hipMemset(p, 0, count * 4).is_success());
                    p.cast::<f32>()
                };
                let alloc_u16 = |count: usize| -> *mut c_void {
                    let mut p: *mut c_void = ptr::null_mut();
                    assert!(hipMalloc(&mut p, count * 2).is_success());
                    assert!(hipMemset(p, 0, count * 2).is_success());
                    p
                };

                let view = EngineStateView {
                    n_gdn,
                    n_attn,
                    gdn_h,
                    gdn_d,
                    gdn_s,
                    gdn_q,
                    gdn_c,
                    n_kv,
                    d_head,
                    max_seq,
                    d_model,
                    use_bf16_kv,
                    gdn_state: alloc_f32(gdn_len),
                    conv_state: alloc_f32(conv_len),
                    k_cache: alloc_u16(k_u16_len),
                    v_cache: alloc_u16(v_u16_len),
                    h_mtp_prev: alloc_f32(mtp_len),
                    x: alloc_f32(x_len),
                    s_main,
                    n_past: 0,
                    gdn_valid: true,
                };

                Self {
                    view,
                    gdn_len,
                    conv_len,
                    mtp_len,
                    x_len,
                    k_u16_len,
                    v_u16_len,
                }
            }
        }

        fn upload_pattern(&mut self, seed: u32, gdn_valid: bool) {
            let gdn_host: Vec<f32> = (0..self.gdn_len)
                .map(|i| f32::from_bits(0x3f80_0000u32 + seed * 1000 + (i as u32)))
                .collect();
            let conv_host: Vec<f32> = (0..self.conv_len)
                .map(|i| f32::from_bits(0x4000_0000u32 + seed * 1000 + (i as u32)))
                .collect();
            let mtp_host: Vec<f32> = (0..self.mtp_len)
                .map(|i| f32::from_bits(0x4040_0000u32 + seed * 1000 + (i as u32)))
                .collect();
            let x_host: Vec<f32> = (0..self.x_len)
                .map(|i| f32::from_bits(0x4080_0000u32 + seed * 1000 + (i as u32)))
                .collect();
            let k_host: Vec<u16> = (0..self.k_u16_len)
                .map(|i| ((seed as usize * 31 + i * 7 + 1) & 0xFFFF) as u16)
                .collect();
            let v_host: Vec<u16> = (0..self.v_u16_len)
                .map(|i| ((seed as usize * 53 + i * 11 + 3) & 0xFFFF) as u16)
                .collect();

            unsafe {
                assert!(hipMemcpy(
                    self.view.gdn_state.cast::<c_void>(),
                    gdn_host.as_ptr().cast::<c_void>(),
                    self.gdn_len * 4,
                    hipMemcpyHostToDevice,
                )
                .is_success());
                assert!(hipMemcpy(
                    self.view.conv_state.cast::<c_void>(),
                    conv_host.as_ptr().cast::<c_void>(),
                    self.conv_len * 4,
                    hipMemcpyHostToDevice,
                )
                .is_success());
                assert!(hipMemcpy(
                    self.view.h_mtp_prev.cast::<c_void>(),
                    mtp_host.as_ptr().cast::<c_void>(),
                    self.mtp_len * 4,
                    hipMemcpyHostToDevice,
                )
                .is_success());
                assert!(hipMemcpy(
                    self.view.x.cast::<c_void>(),
                    x_host.as_ptr().cast::<c_void>(),
                    self.x_len * 4,
                    hipMemcpyHostToDevice,
                )
                .is_success());
                assert!(hipMemcpy(
                    self.view.k_cache,
                    k_host.as_ptr().cast::<c_void>(),
                    self.k_u16_len * 2,
                    hipMemcpyHostToDevice,
                )
                .is_success());
                assert!(hipMemcpy(
                    self.view.v_cache,
                    v_host.as_ptr().cast::<c_void>(),
                    self.v_u16_len * 2,
                    hipMemcpyHostToDevice,
                )
                .is_success());
            }
            self.view.gdn_valid = gdn_valid;
        }

        fn zero_all(&mut self) {
            unsafe {
                assert!(
                    hipMemset(self.view.gdn_state.cast::<c_void>(), 0, self.gdn_len * 4)
                        .is_success()
                );
                assert!(
                    hipMemset(self.view.conv_state.cast::<c_void>(), 0, self.conv_len * 4)
                        .is_success()
                );
                assert!(
                    hipMemset(self.view.h_mtp_prev.cast::<c_void>(), 0, self.mtp_len * 4)
                        .is_success()
                );
                assert!(hipMemset(self.view.x.cast::<c_void>(), 0, self.x_len * 4).is_success());
                assert!(hipMemset(self.view.k_cache, 0, self.k_u16_len * 2).is_success());
                assert!(hipMemset(self.view.v_cache, 0, self.v_u16_len * 2).is_success());
            }
            self.view.n_past = 0;
            self.view.gdn_valid = false;
        }

        fn download_state_bits(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
            let mut gdn = vec![0u32; self.gdn_len];
            let mut conv = vec![0u32; self.conv_len];
            let mut mtp = vec![0u32; self.mtp_len];
            let mut x = vec![0u32; self.x_len];
            unsafe {
                assert!(hipMemcpy(
                    gdn.as_mut_ptr().cast::<c_void>(),
                    self.view.gdn_state.cast_const().cast::<c_void>(),
                    self.gdn_len * 4,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
                assert!(hipMemcpy(
                    conv.as_mut_ptr().cast::<c_void>(),
                    self.view.conv_state.cast_const().cast::<c_void>(),
                    self.conv_len * 4,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
                assert!(hipMemcpy(
                    mtp.as_mut_ptr().cast::<c_void>(),
                    self.view.h_mtp_prev.cast_const().cast::<c_void>(),
                    self.mtp_len * 4,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
                assert!(hipMemcpy(
                    x.as_mut_ptr().cast::<c_void>(),
                    self.view.x.cast_const().cast::<c_void>(),
                    self.x_len * 4,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
            }
            (gdn, conv, mtp, x)
        }

        fn download_kv(&self) -> (Vec<u16>, Vec<u16>) {
            let mut k = vec![0u16; self.k_u16_len];
            let mut v = vec![0u16; self.v_u16_len];
            unsafe {
                assert!(hipMemcpy(
                    k.as_mut_ptr().cast::<c_void>(),
                    self.view.k_cache.cast_const(),
                    self.k_u16_len * 2,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
                assert!(hipMemcpy(
                    v.as_mut_ptr().cast::<c_void>(),
                    self.view.v_cache.cast_const(),
                    self.v_u16_len * 2,
                    hipMemcpyDeviceToHost,
                )
                .is_success());
            }
            (k, v)
        }
    }

    impl Drop for MockGpuEngine {
        fn drop(&mut self) {
            unsafe {
                let _ = hipFree(self.view.gdn_state.cast::<c_void>());
                let _ = hipFree(self.view.conv_state.cast::<c_void>());
                let _ = hipFree(self.view.k_cache);
                let _ = hipFree(self.view.v_cache);
                let _ = hipFree(self.view.h_mtp_prev.cast::<c_void>());
                let _ = hipFree(self.view.x.cast::<c_void>());
                let _ = hipStreamDestroy(self.view.s_main);
            }
        }
    }

    #[test]
    fn test_prompt_cache_save_dedup_fifo_and_restore() {
        let mut mock = MockGpuEngine::new(true);
        let mut cache = PromptCache::with_config(2, 2, 1024 * 1024);

        // Save slot 1 in `snaps`
        mock.upload_pattern(1, true);
        let expected_1 = mock.download_state_bits();
        cache.save(&mock.view, &[10, 20, 30], false);
        assert_eq!(cache.snaps.len(), 1);
        let first_ptr = cache.snaps[0].buf;

        // Deduplicate on identical tokens `[10, 20, 30]` with updated state
        mock.upload_pattern(2, false);
        let expected_2 = mock.download_state_bits();
        cache.save(&mock.view, &[10, 20, 30], false);
        assert_eq!(cache.snaps.len(), 1);
        assert_eq!(cache.snaps[0].buf, first_ptr);
        assert!(!cache.snaps[0].gdn_valid);
        let _ = expected_1;

        // Save slot 2 in `snaps`
        mock.upload_pattern(3, true);
        let expected_3 = mock.download_state_bits();
        cache.save(&mock.view, &[10, 20, 30, 40, 50], false);
        assert_eq!(cache.snaps.len(), 2);

        // Save slot 3 in `snaps`: exceeds `max_snaps = 2`, so `[10, 20, 30]` is FIFO-evicted and its `buf` reused!
        mock.upload_pattern(4, true);
        let expected_4 = mock.download_state_bits();
        cache.save(&mock.view, &[99, 100], false);
        assert_eq!(cache.snaps.len(), 2);
        assert_eq!(cache.snaps[0].tokens, vec![10, 20, 30, 40, 50]);
        assert_eq!(cache.snaps[1].tokens, vec![99, 100]);
        assert_eq!(cache.snaps[1].buf, first_ptr);

        // Save message-boundary snapshot in `msg` pool for `[10, 20, 30, 40, 50, 60]`
        mock.upload_pattern(5, false);
        let expected_5 = mock.download_state_bits();
        cache.save(&mock.view, &[10, 20, 30, 40, 50, 60], true);
        assert_eq!(cache.msg.len(), 1);

        // Zero engine state and restore `[10, 20, 30, 40, 50, 77]` -> should match `[10, 20, 30, 40, 50]` from `snaps`
        mock.zero_all();
        let (reuse, gdn_valid, from_stash) =
            cache.restore(&mut mock.view, &[10, 20, 30, 40, 50, 77]);
        assert_eq!(reuse, 5);
        assert!(gdn_valid);
        assert!(!from_stash);
        assert_eq!(mock.view.n_past, 5);
        assert!(mock.view.gdn_valid);
        assert_eq!(mock.download_state_bits(), expected_3);

        // Restore `[10, 20, 30, 40, 50, 60, 88]` -> should match longer `[10, 20, 30, 40, 50, 60]` from `msg`
        mock.zero_all();
        let (reuse_msg, gdn_valid_msg, from_stash_msg) =
            cache.restore(&mut mock.view, &[10, 20, 30, 40, 50, 60, 88]);
        assert_eq!(reuse_msg, 6);
        assert!(!gdn_valid_msg);
        assert!(!from_stash_msg);
        assert_eq!(mock.view.n_past, 6);
        assert!(!mock.view.gdn_valid);
        assert_eq!(mock.download_state_bits(), expected_5);

        // Restore `[99, 100]` exact length match -> should match `[99, 100]`
        mock.zero_all();
        let (reuse_exact, gdn_valid_exact, from_stash_exact) =
            cache.restore(&mut mock.view, &[99, 100]);
        assert_eq!(reuse_exact, 2);
        assert!(gdn_valid_exact);
        assert!(!from_stash_exact);
        assert_eq!(mock.download_state_bits(), expected_4);
        let _ = expected_2;
    }

    #[test]
    fn test_stash_roundtrip_bf16_and_transposed_v_2d_pitched() {
        // 1. Test with `use_bf16_kv = true` + budget enforcement + FIFO eviction
        {
            let mut mock = MockGpuEngine::new(true);
            let toks_a: Vec<i32> = (0..12).map(|i| 100 + i).collect();
            let toks_b: Vec<i32> = (0..12).map(|i| 200 + i).collect();
            let single_need = state_bytes(&mock.view) + kv_bytes(&mock.view, toks_a.len() as i32);

            // Budget holds only 1 stash entry of length 12
            let mut cache = PromptCache::with_config(4, 4, single_need + 16);

            mock.upload_pattern(10, true);
            let state_a = mock.download_state_bits();
            let (k_a, v_a) = mock.download_kv();
            cache.stash(&mock.view, &toks_a);
            assert_eq!(cache.stashes.len(), 1);
            assert_eq!(cache.stash_used_bytes(), single_need);

            // Stashing a second entry of size `single_need` must evict `toks_a`
            mock.upload_pattern(20, false);
            let state_b = mock.download_state_bits();
            let (k_b, v_b) = mock.download_kv();
            cache.stash(&mock.view, &toks_b);
            assert_eq!(cache.stashes.len(), 1);
            assert_eq!(cache.stashes[0].tokens, toks_b);
            let _ = (state_a, k_a, v_a);

            // Zero engine and restore `toks_b` + extra suffix tokens
            mock.zero_all();
            let mut query = toks_b.clone();
            query.extend_from_slice(&[999, 1000]);
            let (reuse, gdn_valid, from_stash) = cache.restore(&mut mock.view, &query);
            assert_eq!(reuse, toks_b.len() as i32);
            assert!(!gdn_valid);
            assert!(from_stash);
            assert_eq!(mock.download_state_bits(), state_b);

            let (k_restored, v_restored) = mock.download_kv();
            let row_u16 = (mock.view.n_kv * mock.view.d_head) as usize;
            let stride_u16 = (mock.view.max_seq as usize) * row_u16;
            let copied_u16 = toks_b.len() * row_u16;
            for ai in 0..mock.view.n_attn {
                let start = ai * stride_u16;
                assert_eq!(
                    &k_restored[start..start + copied_u16],
                    &k_b[start..start + copied_u16]
                );
                assert_eq!(
                    &v_restored[start..start + copied_u16],
                    &v_b[start..start + copied_u16]
                );
            }
        }

        // 2. Test with `use_bf16_kv = false` (transposed FP16 V cache via 2D pitched copy)
        {
            let mut mock = MockGpuEngine::new(false);
            let l = 11usize;
            let toks: Vec<i32> = (0..l).map(|i| 500 + (i as i32)).collect();
            let mut cache = PromptCache::new();

            mock.upload_pattern(77, true);
            let expected_state = mock.download_state_bits();
            let (expected_k, expected_v) = mock.download_kv();

            cache.stash(&mock.view, &toks);
            assert_eq!(cache.stashes.len(), 1);

            mock.zero_all();
            let mut prompt = toks.clone();
            prompt.push(777);
            let (reuse, gdn_valid, from_stash) = cache.restore(&mut mock.view, &prompt);
            assert_eq!(reuse, l as i32);
            assert!(gdn_valid);
            assert!(from_stash);
            assert_eq!(mock.download_state_bits(), expected_state);

            let (restored_k, restored_v) = mock.download_kv();
            let row_u16 = (mock.view.n_kv * mock.view.d_head) as usize;
            let stride_u16 = (mock.view.max_seq as usize) * row_u16;
            for ai in 0..mock.view.n_attn {
                let start = ai * stride_u16;
                let end = start + l * row_u16;
                assert_eq!(&restored_k[start..end], &expected_k[start..end]);
            }

            // Verify every pitched row of the transposed V cache `[n_attn, d_head, n_kv, max_seq]`
            // for columns `0..vt_round(l)` (16 columns out of pitch `max_seq = 32`).
            let rows = mock.view.n_attn * (mock.view.d_head as usize) * (mock.view.n_kv as usize);
            let pitch_u16 = mock.view.max_seq as usize;
            let vt_cols = vt_round(l as i32) as usize;
            for r in 0..rows {
                let row_start = r * pitch_u16;
                assert_eq!(
                    &restored_v[row_start..row_start + vt_cols],
                    &expected_v[row_start..row_start + vt_cols],
                    "transposed V cache mismatch on row {r}"
                );
                // Columns beyond `vt_cols` should remain untouched (zeroed after `zero_all`)
                assert!(
                    restored_v[row_start + vt_cols..row_start + pitch_u16]
                        .iter()
                        .all(|&v| v == 0),
                    "pitched padding columns were unexpectedly overwritten on row {r}"
                );
            }
        }
    }

    #[test]
    fn test_prefix_cache_trait() {
        let mut pcache = PromptCache::with_config(2, 2, 1024);
        let toks = [1u32, 2, 3, 4, 5];
        let snap = Snapshot {
            pos: 3,
            tokens_hash: compute_tokens_hash(&toks[..3]),
            kv_bytes: 128,
            recurrent_bytes: 256,
        };
        pcache.put(3, snap, SnapReason::EndOfPrompt);

        let matched = pcache.longest(&toks);
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().0, 3);

        let mut standalone = TalusPrefixCache::with_slots(2, 2);
        let h = standalone
            .put_with_tokens(
                &toks[..4],
                Snapshot {
                    pos: 4,
                    tokens_hash: 0,
                    kv_bytes: 256,
                    recurrent_bytes: 512,
                },
                SnapReason::MessageStart,
            )
            .unwrap();
        assert_eq!(standalone.longest(&toks), Some((4, h)));
    }

    #[test]
    fn test_q36_per_layer_prompt_cache_roundtrip() {
        unsafe {
            let mut st: hipStream_t = ptr::null_mut();
            assert!(hipStreamCreateWithFlags(&mut st, hipStreamNonBlocking).is_success());

            let alloc_bytes = |n: usize| -> *mut c_void {
                let mut p: *mut c_void = ptr::null_mut();
                assert!(hipMalloc(&mut p, n).is_success());
                assert!(hipMemset(p, 0, n).is_success());
                p
            };

            let conv0 = alloc_bytes(3 * 8 * 4).cast::<f32>();
            let s0 = alloc_bytes(2 * 4 * 4 * 4).cast::<f32>();
            let kc1 = alloc_bytes(2 * 32 * 8 * 2);
            let vc1 = alloc_bytes(2 * 32 * 8 * 2);
            let vtc1 = alloc_bytes(2 * 8 * 32 * 2);
            let hlast = alloc_bytes(16 * 4).cast::<f32>();
            let gctl = alloc_bytes(12).cast::<i32>();

            let mut view = Q36LayerStateView {
                ssm_d_conv: 4,
                ssm_conv_channels: 8,
                ssm_v_heads: 2,
                ssm_state_size: 4,
                n_head_kv: 2,
                head_dim: 8,
                n_embd: 16,
                ctx: 32,
                vt_ctx: 32,
                has_mtp: true,
                conv: vec![conv0, ptr::null_mut()],
                s: vec![s0, ptr::null_mut()],
                kc: vec![ptr::null_mut(), kc1],
                vc: vec![ptr::null_mut(), vc1],
                vtc: vec![ptr::null_mut(), vtc1],
                hlast,
                gctl,
                gdn_pending: 7,
                pos: 0,
                st,
                dfl_k_ring: Vec::new(),
                dfl_v_ring: Vec::new(),
                dfl_ring: 0,
                dfl_keep: 0,
                dfl_kv_dim: 0,
                dfl_block: 0,
                dctx_lo: 0,
                dctx_hi: 0,
            };

            assert_eq!(
                q36_state_bytes(&view),
                (3 * 8 * 4) + (2 * 4 * 4 * 4) + (16 * 4) + 16
            );
            assert_eq!(q36_kv_bytes(&view, 10, 0), (10 * 2 + 16) * (2 * 8 * 2));

            let mut qcache = Q36PromptCache::new();
            let toks = vec![1, 2, 3, 4, 5, 6];
            qcache.save(&mut view, &toks, false);

            view.gdn_pending = 0;
            view.pos = 0;
            let reused = qcache.restore(&mut view, &[1, 2, 3, 4, 5, 6, 7], &toks);
            assert_eq!(reused, 6);
            assert_eq!(view.pos, 6);
            assert_eq!(view.gdn_pending, 7);

            drop(qcache);
            let _ = hipFree(conv0.cast::<c_void>());
            let _ = hipFree(s0.cast::<c_void>());
            let _ = hipFree(kc1);
            let _ = hipFree(vc1);
            let _ = hipFree(vtc1);
            let _ = hipFree(hlast.cast::<c_void>());
            let _ = hipFree(gctl.cast::<c_void>());
            let _ = hipStreamDestroy(st);
        }
    }
}

#[cfg(test)]
mod disk_tests;
