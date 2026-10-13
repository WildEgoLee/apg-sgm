#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/result.hpp"

namespace apg {

class StereoMatcher {
public:
    explicit StereoMatcher(PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced));

    void set_config(const PipelineConfig& cfg) { cfg_ = cfg; }
    const PipelineConfig& config() const { return cfg_; }

    // Modern public API (isolates internal pipeline buffers)
    bool compute(const Image8& left, const Image8& right, StereoResult& result,
                 PipelineStats* stats = nullptr) const;

    // Supported / Legacy API (retains source compatibility with P3)
    bool compute(const Image8& left, const Image8& right, PipelineBuffers& out,
                 PipelineStats* stats = nullptr) const;

    const std::string& last_error() const { return last_error_; }

private:
    bool compute_impl(const Image8& left, const Image8& right, PipelineBuffers& out,
                      PipelineStats* stats) const;

    PipelineConfig cfg_;
    mutable std::string last_error_;
};

} // namespace apg
