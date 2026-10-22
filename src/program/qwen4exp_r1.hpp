// src/program/qwen4exp_r1.hpp - qwen4exp-r1: generate.cpp's calls into the Rust expert source / PLE table reader factory.
#pragma once

#include "talus_qwen4exp_kernels.h"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/ngram.hpp"

#include <string>

namespace qwen4exp_r1 {

/// The registered Rust factory, or null (not registered, or TALUS_FN_R1=0).
const tk_qwen4exp_r1_rust_hooks_t* hooks();

/// `src` (not yet opened) adopts a Rust source for the GGUF shards of `pack_dir` / `gguf`; false (and one line why)
/// when Rust declines the run's mode or no factory is registered - the C++ source then runs as before.
bool adopt_source(strata::core::FileExpertSource& src, const std::string& pack_dir, const std::string& gguf);

/// `table` (closed) adopts a Rust PLE table reader; false when none is registered.
bool adopt_ple(strata::kernels::PleTable& table);

/// Rust frees what it allocated for the C++ objects (they borrowed it while hooks() was live); call once they are gone.
void release_rust_state();

}  // namespace qwen4exp_r1
