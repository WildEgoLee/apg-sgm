#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class StereoMatcher {
public:
    explicit StereoMatcher(PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced));

    void set_config(const PipelineConfig& cfg) { cfg_ = cfg; }
    const PipelineConfig& config() const { return cfg_; }

    bool compute(const Image8& left, const Image8& right, PipelineBuffers& out) const;
    const std::string& last_error() const { return last_error_; }

private:
    PipelineConfig cfg_;
    mutable std::string last_error_;
};

} // namespace apg
