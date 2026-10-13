#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"
#include "apg_sgm/image_view.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/result.hpp"
#include "apg_sgm/status.hpp"
#include "apg_sgm/workspace.hpp"

namespace apg {

Status validate_config(const PipelineConfig& cfg);
Status validate_inputs(const ImageView8& left, const ImageView8& right);
inline Status validate_inputs(const Image8& left, const Image8& right) {
    return validate_inputs(ImageView8(left), ImageView8(right));
}
inline Status validate_inputs(const StereoInputView& input) {
    return validate_inputs(input.left, input.right);
}

size_t estimate_workspace_bytes(int width, int height, const PipelineConfig& cfg);

class StereoMatcher {
public:
    explicit StereoMatcher(PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced));

    void set_config(const PipelineConfig& cfg) { cfg_ = cfg; }
    const PipelineConfig& config() const { return cfg_; }

    // Modern structured Status API (with ImageView8 & explicit workspace)
    Status compute_status(const ImageView8& left, const ImageView8& right, StereoResult& result,
                          StereoWorkspace& workspace, PipelineStats* stats = nullptr) const;

    // Modern structured Status API (with ImageView8 & default workspace)
    Status compute_status(const ImageView8& left, const ImageView8& right, StereoResult& result,
                          PipelineStats* stats = nullptr) const;

    // StereoInputView Status overloads
    Status compute_status(const StereoInputView& input, StereoResult& result,
                          StereoWorkspace& workspace, PipelineStats* stats = nullptr) const {
        return compute_status(input.left, input.right, result, workspace, stats);
    }
    Status compute_status(const StereoInputView& input, StereoResult& result,
                          PipelineStats* stats = nullptr) const {
        return compute_status(input.left, input.right, result, stats);
    }

    // Image8 Status overloads
    Status compute_status(const Image8& left, const Image8& right, StereoResult& result,
                          StereoWorkspace& workspace, PipelineStats* stats = nullptr) const {
        return compute_status(ImageView8(left), ImageView8(right), result, workspace, stats);
    }
    Status compute_status(const Image8& left, const Image8& right, StereoResult& result,
                          PipelineStats* stats = nullptr) const {
        return compute_status(ImageView8(left), ImageView8(right), result, stats);
    }

    // Modern public API (with ImageView8 & explicit workspace)
    bool compute(const ImageView8& left, const ImageView8& right, StereoResult& result,
                 StereoWorkspace& workspace, PipelineStats* stats = nullptr) const;

    // Modern public API (with ImageView8 & default workspace)
    bool compute(const ImageView8& left, const ImageView8& right, StereoResult& result,
                 PipelineStats* stats = nullptr) const;

    // StereoInputView bool overloads
    bool compute(const StereoInputView& input, StereoResult& result,
                 StereoWorkspace& workspace, PipelineStats* stats = nullptr) const {
        return compute(input.left, input.right, result, workspace, stats);
    }
    bool compute(const StereoInputView& input, StereoResult& result,
                 PipelineStats* stats = nullptr) const {
        return compute(input.left, input.right, result, stats);
    }

    // Image8 bool overloads
    bool compute(const Image8& left, const Image8& right, StereoResult& result,
                 StereoWorkspace& workspace, PipelineStats* stats = nullptr) const {
        return compute(ImageView8(left), ImageView8(right), result, workspace, stats);
    }
    bool compute(const Image8& left, const Image8& right, StereoResult& result,
                 PipelineStats* stats = nullptr) const {
        return compute(ImageView8(left), ImageView8(right), result, stats);
    }

    // Supported / Legacy API (retains source compatibility with P3)
    bool compute(const Image8& left, const Image8& right, PipelineBuffers& out,
                 PipelineStats* stats = nullptr) const;

    const std::string& last_error() const { return last_error_; }

private:
    Status compute_impl(const ImageView8& left, const ImageView8& right, PipelineBuffers& out,
                        PipelineStats* stats, int num_threads = 0) const;

    PipelineConfig cfg_;
    mutable std::string last_error_;
};

} // namespace apg
