#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg::detail {

// Portable scalar Refiner implementation (identical to 56554b4)
void refine_scalar(const PipelineConfig& cfg, PipelineBuffers& buf);

// Dedicated AVX2 Refiner implementation with per-pixel invariant hoisting
void refine_avx2(const PipelineConfig& cfg, PipelineBuffers& buf);

} // namespace apg::detail
