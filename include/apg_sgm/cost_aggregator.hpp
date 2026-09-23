#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"
#include "apg_sgm/packed_volume.hpp"

namespace apg {

class CostAggregator {
public:
    void aggregate(const PipelineConfig& cfg, PipelineBuffers& buf) const;

    void aggregate_packed(const PipelineConfig& cfg,
                          const Image8& left_gray,
                          PackedCostVolume16& packed_cost) const;

private:
    void build_cross_arms(const Image8& gray, const AggregationParams& p,
                          std::vector<int16_t>& left, std::vector<int16_t>& right,
                          std::vector<int16_t>& up, std::vector<int16_t>& down) const;

    void aggregate_hv(PipelineBuffers& buf,
                      const std::vector<int16_t>& L, const std::vector<int16_t>& R,
                      const std::vector<int16_t>& U, const std::vector<int16_t>& D) const;

    void aggregate_hv_packed(PackedCostVolume16& packed_cost,
                             PackedCostVolume16& tmp,
                             const std::vector<int16_t>& L, const std::vector<int16_t>& R,
                             const std::vector<int16_t>& U, const std::vector<int16_t>& D) const;
};

} // namespace apg
