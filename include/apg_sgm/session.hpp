#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/execution.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/result.hpp"
#include "apg_sgm/status.hpp"
#include "apg_sgm/workspace.hpp"

#include <chrono>
#include <string>
#include <utility>

namespace apg {

class StereoSession {
public:
    explicit StereoSession(PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced),
                           ExecutionOptions opt = {})
        : cfg_(std::move(cfg)),
          workspace_(opt),
          matcher_(cfg_) {}

    // Modern structured Status stream processing API
    Status process_status(const Image8& left, const Image8& right, StereoResult& result,
                          PipelineStats* stats = nullptr) {
        last_error_.clear();
        auto t0 = std::chrono::steady_clock::now();
        Status s = matcher_.compute_status(left, right, result, workspace_, stats);
        auto t1 = std::chrono::steady_clock::now();

        if (!s.ok()) {
            last_error_ = s.message;
            return s;
        }

        frame_count_++;
        total_time_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        return Status::Ok();
    }

    // Modern public stream processing API (bool return)
    bool process(const Image8& left, const Image8& right, StereoResult& result,
                 PipelineStats* stats = nullptr) {
        Status s = process_status(left, right, result, stats);
        return s.ok();
    }

    void reset() {
        workspace_.release();
        frame_count_ = 0;
        total_time_ms_ = 0.0;
        last_error_.clear();
    }

    void set_config(const PipelineConfig& cfg) {
        cfg_ = cfg;
        matcher_.set_config(cfg_);
    }
    const PipelineConfig& config() const { return cfg_; }

    void set_options(const ExecutionOptions& opt) {
        workspace_.set_options(opt);
    }
    const ExecutionOptions& options() const { return workspace_.options(); }

    size_t frame_count() const { return frame_count_; }
    double total_time_ms() const { return total_time_ms_; }
    double average_time_ms() const {
        return frame_count_ > 0 ? (total_time_ms_ / static_cast<double>(frame_count_)) : 0.0;
    }
    size_t allocated_bytes() const { return workspace_.allocated_bytes(); }
    const std::string& last_error() const { return last_error_; }

    StereoWorkspace& workspace() { return workspace_; }
    const StereoWorkspace& workspace() const { return workspace_; }

private:
    PipelineConfig cfg_;
    StereoWorkspace workspace_;
    StereoMatcher matcher_;
    size_t frame_count_ = 0;
    double total_time_ms_ = 0.0;
    std::string last_error_;
};

} // namespace apg
