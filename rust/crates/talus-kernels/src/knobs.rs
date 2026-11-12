//! Process-environment defaults for the kernel library, owned by the Rust host  S3).
//!
//! The kernel library reads its switches with `getenv` (mostly once, into function-local statics). Each engine sets
//! its defaults before the first read, and "a variable the user exported always wins": a default is applied only when
//! the variable is unset. Before S3 some of these defaults were set from C++ (`ensure_q36_env` in
//! `src/qwen35/q36_kernels.cu`, which ran inside the first `tk_q36_mmq_built` / `tk_q36_mmq_supported_fits` /
//! `tk_q36_q8_0_pack_should` call). [`q36_env`] now applies exactly those values, and the Rust wrappers call it
//! immediately before each of those three entry points, so the library sees the same environment at the same moment.
//!
//! `TALUS_KNOBS_LOG=1` prints every default this module applies (one line per variable, stderr).

use std::sync::Once;

/// A named set of defaults (`(variable, value)`), applied with "unset only" semantics.
pub type KnobTable = &'static [(&'static str, &'static str)];

/// The values `ensure_q36_env` set in C++ (q36_kernels.cu, before S3): eager module loading (CUDA builds;
/// the HIP runtime does not read it), the packed Q8_0 decode layout, and the K-quant MMQ prompt kernels.
pub const Q36_ENV: KnobTable = &[
    ("CUDA_MODULE_LOADING", "EAGER"),
    ("STRATA_Q8_PACKED", "1"),
    ("STRATA_MMQ_KQ", "1"),
];

/// GLM-5.3-Flash defaults for the shared kernels (glm/ffi.rs, applied before `tk_ctx_create`): K-quant MMQ for the
/// Q4_K / Q6_K dense projections, no hipBLAS / hipBLASLt handles (GLM never calls them; saves their workspace near the
/// memory floor), and the fp16 routed-expert kernels (`STRATA_PF_W16_MOE=0` = kill switch).
pub const GLM_DEFAULTS: KnobTable = &[
    ("STRATA_MMQ_KQ", "1"),
    ("STRATA_PREFILL_NO_BLAS", "1"),
    ("STRATA_PF_W16_MOE", "1"),
];

/// Laguna engine default (laguna/engine.rs `Engine::new`): the C++ engine's mat-vec family (native_q*_k_mmvq /
/// native_q8_0_mmvq), not the wave32 kernels. The serve path sets [`LAGUNA_SERVE_DEFAULTS`] first.
pub const LAGUNA_ENGINE_DEFAULTS: KnobTable = &[("STRATA_MMVQ_W32", "0")];

/// Laguna serve-mode defaults (talus-cli): the wave32 kernels, and the exact tile kernels for verify windows
/// from 9 columns (`STRATA_MMVQ_XT=0`: off).
pub const LAGUNA_SERVE_DEFAULTS: KnobTable = &[("STRATA_MMVQ_W32", "1"), ("STRATA_MMVQ_XT", "9")];

fn log_on() -> bool {
    std::env::var("TALUS_KNOBS_LOG")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Applies `table` with "unset only" semantics; returns how many variables it set.
pub fn apply(what: &str, table: KnobTable) -> usize {
    let mut n = 0;
    for &(k, v) in table {
        if std::env::var_os(k).is_none() {
            // SAFETY (edition 2024 semantics): called on the engine's setup thread before the kernel library reads
            // the variable; the C side only reads the environment.
            #[allow(unused_unsafe)]
            unsafe {
                std::env::set_var(k, v);
            }
            n += 1;
            if log_on() {
                eprintln!("talus knobs: {what}: {k}={v}");
            }
        }
    }
    n
}

/// Applies [`Q36_ENV`] once per process. Call it immediately before `tk_q36_mmq_built`,
/// `tk_q36_mmq_supported_fits` and `tk_q36_q8_0_pack_should` (the C entry points that used to run `ensure_q36_env`).
pub fn q36_env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        apply("q36", Q36_ENV);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test touches the process environment (tests run in parallel threads of one process).
    #[test]
    fn unset_only_and_once() {
        let k = "TALUS_KNOBS_TEST_VAR_S3";
        std::env::remove_var(k);
        static T: KnobTable = &[("TALUS_KNOBS_TEST_VAR_S3", "7")];
        assert_eq!(apply("t", T), 1);
        assert_eq!(std::env::var(k).unwrap(), "7");
        std::env::set_var(k, "user");
        assert_eq!(apply("t", T), 0);
        assert_eq!(std::env::var(k).unwrap(), "user");
        std::env::remove_var(k);
    }

    #[test]
    fn tables_match_the_cpp_values() {
        assert_eq!(
            Q36_ENV,
            &[
                ("CUDA_MODULE_LOADING", "EAGER"),
                ("STRATA_Q8_PACKED", "1"),
                ("STRATA_MMQ_KQ", "1")
            ]
        );
        assert!(GLM_DEFAULTS
            .iter()
            .any(|&(k, v)| k == "STRATA_PREFILL_NO_BLAS" && v == "1"));
    }
}
