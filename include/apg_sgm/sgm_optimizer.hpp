#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class SgmOptimizer {
public:
    void optimize(const PipelineConfig& cfg, PipelineBuffers& buf) const;

    void winner_take_all(const PipelineConfig& cfg, const CostVolume32& vol,
                         const SearchRange& range, Image32f& disp_out,
                         Image32f* best_cost = nullptr,
                         Image32f* second_cost = nullptr) const;

    void winner_take_all(const PipelineConfig& cfg, const CostVolume& vol,
                         const SearchRange& range, Image32f& disp_out,
                         Image32f* best_cost = nullptr,
                         Image32f* second_cost = nullptr) const;
    void aggregate_path(const PipelineConfig& cfg, const Image8& gray,
                        const CostVolume& base, CostVolume32& acc, int dx, int dy) const;
};

} // namespace apg
