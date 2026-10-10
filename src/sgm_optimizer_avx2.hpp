#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/packed_volume.hpp"

#include "sgm_common.hpp"

namespace apg {
namespace detail {

void aggregate_path_packed_avx2(const PipelineConfig& cfg, const Image8& gray,
                                const PackedCostVolume16& base,
                                PackedCostVolume16& acc, int dx, int dy,
                                AccumulateMode mode = AccumulateMode::Add);

void aggregate_path_packed_avx2(const PipelineConfig& cfg, const Image8& gray,
                                const PackedCostVolume16& base,
                                PackedCostVolume32& acc, int dx, int dy,
                                AccumulateMode mode = AccumulateMode::Add);

void winner_take_all_packed_avx2(const PipelineConfig& cfg,
                                 const PackedCostVolume16& vol,
                                 Image32f& disp_out,
                                 Image32f* best_cost,
                                 Image32f* second_cost);

void winner_take_all_packed_avx2(const PipelineConfig& cfg,
                                 const PackedCostVolume32& vol,
                                 Image32f& disp_out,
                                 Image32f* best_cost,
                                 Image32f* second_cost);

} // namespace detail
} // namespace apg
