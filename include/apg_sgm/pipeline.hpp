#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/result.hpp"
#include "apg_sgm/status.hpp"
#include "apg_sgm/workspace.hpp"

namespace apg {

Status validate_config(const PipelineConfig& cfg);
Status validate_inputs(const Image8& left, const Image8& right);

class StereoMatcher {
public:
    explicit StereoMatcher(PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced));

    void set_config(const PipelineConfig& cfg) { cfg_ = cfg; }
    const PipelineConfig& config() const { return cfg_; }

    // Modern structured Status API (with explicit workspace)
    Status compute_status(const Image8& left, const Image8& right, StereoResult& result,
                          StereoWorkspace& workspace, PipelineStats* stats = nullptr) const;

    // Modern structured Status API (default workspace with ReleasePerFrame)
    Status compute_status(const Image8& left, const Image8& right, StereoResult& result,
                          PipelineStats* stats = nullptr) const;

    // Modern public API (with explicit workspace)
    bool compute(const Image8& left, const Image8& right, StereoResult& result,
                 StereoWorkspace& workspace, PipelineStats* stats = nullptr) const;

    // Modern public API (bool return)
    bool compute(const Image8& left, const Image8& right, StereoResult& result,
                 PipelineStats* stats = nullptr) const;

    // Supported / Legacy API (retains source compatibility with P3)
    bool compute(const Image8& left, const Image8& right, PipelineBuffers& out,
                 PipelineStats* stats = nullptr) const;

    const std::string& last_error() const { return last_error_; }

private:
    Status compute_impl(const Image8& left, const Image8& right, PipelineBuffers& out,
                        PipelineStats* stats) const;

    PipelineConfig cfg_;
    mutable std::string last_error_;
};

} // namespace apg
