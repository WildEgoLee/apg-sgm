#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/buffers.hpp"
#include "apg_sgm/packed_volume.hpp"

namespace apg::detail {

void compute_volume_packed_avx2(
    const PipelineConfig& cfg,
    const PipelineBuffers& buf,
    PackedCostVolume16& packed_cost);

void build_symmetric_census9x7_avx2(
    const Image8& gray,
    std::vector<uint32_t>& c32);

} // namespace apg::detail
