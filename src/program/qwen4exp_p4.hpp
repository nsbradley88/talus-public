// src/program/qwen4exp_p4.hpp - qwen4exp Piece 4: what the serve / generate loop hands the Rust Verifier + MTP host plane
// (rust/crates/talus-model/src/qwen4exp/verify_mtp.rs).  generate.cpp fills one of these per loop (its qwen4exp-p4
// block); src/program/qwen4exp_p4.cpp reads it for the tk_qwen4exp_p4_* entry points.
#pragma once

#include "talus_qwen4exp_kernels.h"

#include "strata/core/mtp.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"

#include <cstdint>
#include <functional>

struct tk_qwen4exp_p4_hooks {
    strata::core::Verifier* ver = nullptr;
    strata::core::MtpDrafter* mtp = nullptr;
    strata::core::SessionState* ss = nullptr;
    bool serve = false;
    bool use_mtp = false;
    const bool* use_mtp_live = nullptr;   // when set, read at each call (the generate loop decides it in spec_init)
    bool mtp_on() const { return use_mtp_live != nullptr ? *use_mtp_live : use_mtp; }
    int spec_corrupt = 0;
    bool commit_async = false;
    int* hist_n = nullptr;
    int32_t* d_hist = nullptr;
    int hist_cap = 0;
    strata::core::PoolMultiFn pool = nullptr;
    void* pool_user = nullptr;
    // the generate-local work (TK_P4_HOOK_*); an empty one is a no-op returning 1
    std::function<int()> pre_window;
    std::function<int(int64_t, int)> ensure_kv;
    std::function<void()> window_started;
    std::function<void()> post_window;
    std::function<int()> first_verified;   // 1: this was the first verified window
    std::function<void(int64_t)> commit_begin;
    std::function<int()> join_adapt;
    std::function<void()> commit_done;
    std::function<void()> mark_nonfinite;
    std::function<void(int)> draft_done;
    std::function<const char*()> drive_failed;
};
