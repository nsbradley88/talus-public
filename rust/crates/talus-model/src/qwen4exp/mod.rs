//! Flash-Next (`qwen4exp`) Rust host orchestrator module.

pub mod cli;
pub mod conv_state;
pub mod decode;
pub mod engine;
pub mod ffi;
pub mod kld_flags;
pub mod prefill;
pub mod skeleton;
pub mod spec;
pub mod state;
pub mod verify_mtp;
pub mod weights;

pub use cli::{
    classify_serve_line, has_pack_flag, is_qwen4exp_invocation, is_qwen4exp_model,
    parse_fn_dump_compare_args, parse_i64_list, use_cpp_engine, use_rust_engine, ServeCommandKind,
    ARCH_NAME,
};
pub use conv_state::{CheckpointMeta, Piece5ConvState};
pub use decode::Piece3Decode;
pub use engine::{run_qwen4exp_cli, Qwen4ExpDriverState};
pub use ffi::{
    Qwen4ExpKernels, TkQwen4ExpConvStateSizes, TkQwen4ExpKldReport, TkQwen4ExpPrefillStats,
    TkQwen4ExpRustDriver, TkQwen4ExpWeightsInfo,
};
pub use kld_flags::{
    classify_niche_flag, compute_sparse_topk_kld, NicheFlagDisposition, Piece6KldFlags,
};
pub use prefill::{equal_chunk, Piece2Prefill};
pub use skeleton::{
    fnv1a64_bytes, BlockDeviceBuffers, GdnDeviceBuffers, GdnLayerState, LayerDumpComparison,
    LayerDumpEntry, LayerDumpHarness, LayerDumpHeader, MoeDeviceBuffers, PleStateLayout,
    QsaDeviceBuffers, QsaLayerState, Qwen4ExpArchSeam, Qwen4ExpDeviceBuffers, Qwen4ExpKvGdnState,
    Qwen4ExpModelConfig, Qwen4ExpRustEngine, Qwen4ExpScratchArena, Qwen4ExpStreams,
    TensorDiffReport, LAYER_DUMP_MAGIC,
};
pub use spec::{parse_logit_bias, DraftPick, DraftPolicy, SuffixDrafter};
pub use state::{compute_prompt_cuts, message_checkpoint_boundary, PromptCutPlan};
pub use verify_mtp::Piece4VerifyMtp;
pub use weights::{ExpertCacheLayout, Piece1Weights};
