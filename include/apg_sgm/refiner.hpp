#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class Refiner {
public:
    void refine(const PipelineConfig& cfg, PipelineBuffers& buf) const;
};

} // namespace apg
