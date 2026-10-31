//! ModelAdapter implementations and architecture modules.
//!
//! Includes the Qwen3.5 / Qwen3.6 model & engine adapter (`qwen35`, `qwen35moe`).

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::collapsible_if,
    clippy::field_reassign_with_default,
    clippy::identity_op,
    clippy::manual_checked_ops,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    clippy::manual_range_contains,
    clippy::missing_safety_doc,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::ptr_eq,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast
)]

pub mod glm;
pub mod laguna;
pub mod mimo;
pub mod qwen35;
pub mod qwen4exp;
pub mod tail_prune;

pub use laguna::{
    Drafter as LagunaDrafter, Engine as LagunaEngine, LagunaCatalog, LagunaModel,
    Options as LagunaOptions, Sampling as LagunaSampling, SessionHeader as LagunaSessionHeader,
};
pub use qwen35::*;
pub use talus_core_api::ModelAdapter;

pub const CRATE_NAME: &str = "talus-model";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_identity() {
        assert_eq!(CRATE_NAME, "talus-model");
    }
}
