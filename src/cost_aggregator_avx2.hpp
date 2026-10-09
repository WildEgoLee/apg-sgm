#pragma once

#include "apg_sgm/packed_volume.hpp"
#include "cost_aggregator_internal.hpp"

#include <vector>
#include <cstdint>

namespace apg {
namespace detail {

void aggregate_hv_packed_avx2(PackedCostVolume16& packed_cost,
                              PackedCrossTmp& tmp,
                              const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                              const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm);

void aggregate_hv_packed_avx2(PackedCostVolume16& packed_cost,
                              PackedCostVolume16& tmp,
                              const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                              const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm);

} // namespace detail
} // namespace apg
