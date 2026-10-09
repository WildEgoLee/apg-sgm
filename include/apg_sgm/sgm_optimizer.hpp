#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"
#include "apg_sgm/packed_volume.hpp"

namespace apg {

inline bool can_use_u16_sgm_accumulator(const PipelineConfig& cfg, int npath) {
    const uint64_t p2_max = std::max<uint64_t>(
        static_cast<uint64_t>(cfg.sgm.P2_base),
        static_cast<uint64_t>(cfg.sgm.P1) + 1);

    const uint64_t path_max = static_cast<uint64_t>(cfg.cost.cost_max) + p2_max;
    const uint64_t total_max = static_cast<uint64_t>(npath) * path_max;

    return total_max < kInvalidCost;
}

class SgmOptimizer {
public:
    void optimize(const PipelineConfig& cfg, PipelineBuffers& buf) const;
    void optimize_packed(const PipelineConfig& cfg, const Image8& gray,
                         const PackedCostVolume16& packed_cost,
                         PackedCostVolume16& packed_sgm16) const;
    void optimize_packed(const PipelineConfig& cfg, const Image8& gray,
                         const PackedCostVolume16& packed_cost,
                         PackedCostVolume32& packed_cost32) const;

    void winner_take_all(const PipelineConfig& cfg, const CostVolume32& vol,
                         const SearchRange& range, Image32f& disp_out,
                         Image32f* best_cost = nullptr,
                         Image32f* second_cost = nullptr) const;

    void winner_take_all(const PipelineConfig& cfg, const CostVolume& vol,
                         const SearchRange& range, Image32f& disp_out,
                         Image32f* best_cost = nullptr,
                         Image32f* second_cost = nullptr) const;

    void winner_take_all_packed(const PipelineConfig& cfg,
                                const PackedCostVolume32& vol,
                                Image32f& disp_out,
                                Image32f* best_cost = nullptr,
                                Image32f* second_cost = nullptr) const;

    void winner_take_all_packed(const PipelineConfig& cfg,
                                const PackedCostVolume16& vol,
                                Image32f& disp_out,
                                Image32f* best_cost = nullptr,
                                Image32f* second_cost = nullptr) const;

    void aggregate_path(const PipelineConfig& cfg, const Image8& gray,
                        const CostVolume& base, CostVolume32& acc, int dx, int dy) const;

    void aggregate_path_packed(const PipelineConfig& cfg, const Image8& gray,
                               const PackedCostVolume16& base,
                               PackedCostVolume16& acc, int dx, int dy) const;
    void aggregate_path_packed(const PipelineConfig& cfg, const Image8& gray,
                               const PackedCostVolume16& base,
                               PackedCostVolume32& acc, int dx, int dy) const;
};

} // namespace apg
