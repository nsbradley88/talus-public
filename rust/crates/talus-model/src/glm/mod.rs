//! GLM-5.3-Flash (`glm5-next`) model adapter: 34 KDA linear-attention + 11 MLA/DSA sparse-attention layers with
//! mHC residual streams, 288-expert MoE (IQ2_S/IQ3_S), and a NextN (MTP) block. Kernels: the tuned GLM HIP kernels
//! plus `src/glm/glm_kernels.cu` (C ABI `include/talus_glm_kernels.h`) in libtalus_kernels.so.

pub mod cli;
pub mod engine;
pub mod ffi;
pub mod sample;
pub mod spec;
pub mod weights;

pub use engine::{GlmConfig, GlmEngine, TMAX};
pub use weights::{GlmHparams, GlmWeights, ARCH_NAME};

/// `general.architecture == "glm5-next"` (any shard of a split GGUF).
pub fn is_glm_gguf(path: &std::path::Path) -> bool {
    match talus_gguf::GgufModel::open(path) {
        Ok(m) => m.meta_shard().get_str("general.architecture") == Some(ARCH_NAME),
        Err(_) => false,
    }
}
