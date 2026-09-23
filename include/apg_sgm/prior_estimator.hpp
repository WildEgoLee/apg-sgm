#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class PriorEstimator {
public:
    void estimate(const PipelineConfig& cfg, PipelineBuffers& buf) const;

    std::vector<SupportMatch> extract_supports(const PipelineConfig& cfg,
                                               const PipelineBuffers& buf) const;

    void interpolate_prior(const std::vector<SupportMatch>& supports,
                           PipelineBuffers& buf) const;

    void apply_search_range(const PipelineConfig& cfg, PipelineBuffers& buf) const;
};

} // namespace apg
