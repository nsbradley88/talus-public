// src/qwen35/q36_adraft.hpp - Talus (talus-next, lever 5): the device side of the adaptive draft length.
//
// After each MTP draft step the engine records two numbers of the draft distribution q (the draft head's row at the
// request's temperature, restricted to the min-p support exactly as k_lse / k_argmax_samp draw from it; greedy
// requests: softmax at T 1, the full row):
//   conf[step] = q(the drafted token)
//   top[step]  = max q   (a property of q alone: known before the draft is drawn)
// The host's length policy (q36_adraft_policy.hpp) stops a window on `top` only: the rule then never looks at the
// drawn draft it is deciding about, so rejection sampling stays exact in distribution.  `conf` is for the trace.
#pragma once

#include <cstdint>

namespace talus::q36 {

/// conf / top of draft step `step` (writes conf[step], top[step]) from the draft head's logits row `x` of `n`
/// entries; *tau the request's temperature (<= 0: greedy), *lmin ln(min_p) (-inf: none), *pick the drafted token
/// (a vocabulary id: mapped through `map` when the draft vocabulary is on, map == nullptr: the identity).
/// One block of 1024 on `stream`; graph-capturable.
/// Sampled requests (tau > 0): the support and log-sum-exp from draft_support (lse[step], thr[step]) when given.
void draft_conf(const float* x, int n, const float* tau, const float* lmin, const int32_t* pick, const int32_t* map,
                float* conf, float* top, int step, void* stream, const float* lse = nullptr, const float* thr = nullptr);

/// Sampled drafts (STRATA_DRAFT_NUCLEUS): the draft distribution's support mirrors the request's sampler chain as
/// make_dist() applies it - top-k on the logits (k <= 0: 64), the temperature, top-p over the top-k set, min-p
/// relative to the max - so less of q falls where the target's p is zero.  *thr = the support's threshold on x / tau
/// (the float test k_argmax_samp and the host's q use), *lse = log sum exp(x / tau) over it.  tau <= 0: *lse = 0.
/// The bounds come from a bisection over the candidates within 30 nats of the max (the test stays the threshold
/// the host reads, so any rounding there leaves the rejection sampling exact).  One block of 1024.
void draft_support(const float* x, int n, const float* tau, const float* lmin, const float* top_p, const int* top_k,
                   float* lse, float* thr, void* stream);

}  // namespace talus::q36
