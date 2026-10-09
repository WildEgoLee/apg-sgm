#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/packed_volume.hpp"

namespace apg {
namespace detail {

void aggregate_path_packed_avx2(const PipelineConfig& cfg, const Image8& gray,
                                const PackedCostVolume16& base,
                                PackedCostVolume32& acc, int dx, int dy);

} // namespace detail
} // namespace apg
