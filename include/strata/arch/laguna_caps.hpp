// include/strata/arch/laguna_caps.hpp - poolside Laguna-S-2.1 (`laguna`): the arch name and the shape facts this
// build relies on.  The description itself comes from the GGUF (strata/core/model_desc.hpp, `laguna_detail`).
//
// Status: the description, the tensor tables and the load-time tensor check exist; the forward pass does
// not.  `laguna_detail::capability` refuses every run with that reason until the kernels land.
//
// Header-only and std-only, so CPU tools can include it.
#pragma once

#include <cstdint>

namespace strata::arch::laguna {

inline constexpr const char* kArch = "laguna";

// the layer pattern llama.cpp derives when the file carries none (load_swa_pattern(ml, 4, dense_first = true)):
// full attention at layer % kFullEvery == 0, sliding-window attention elsewhere
inline constexpr int64_t kFullEvery = 4;

// the expert gating codes GGUF uses (llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_*)
inline constexpr int64_t kGatingSoftmax = 1;
inline constexpr int64_t kGatingSigmoid = 2;

}  // namespace strata::arch::laguna
