#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/buffers.hpp"
#include "apg_sgm/packed_volume.hpp"

namespace apg::detail {

void compute_volume_packed_avx2(
    const PipelineConfig& cfg,
    const PipelineBuffers& buf,
    PackedCostVolume16& packed_cost);

} // namespace apg::detail
