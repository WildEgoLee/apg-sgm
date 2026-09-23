#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class ConfidenceEstimator {
public:
    void estimate(const PipelineConfig& cfg, PipelineBuffers& buf,
                  const Image32f& best_cost, const Image32f& second_cost) const;
};

} // namespace apg
