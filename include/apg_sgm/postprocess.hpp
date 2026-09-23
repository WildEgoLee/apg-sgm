#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class PostProcessor {
public:
    void left_right_check(const PipelineConfig& cfg, PipelineBuffers& buf) const;
    void fill_holes(const PipelineConfig& cfg, PipelineBuffers& buf) const;
    void median(const PipelineConfig& cfg, PipelineBuffers& buf) const;
};

} // namespace apg
