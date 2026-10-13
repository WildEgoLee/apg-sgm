#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/execution.hpp"

#include <cstddef>

namespace apg {

class StereoWorkspace {
public:
    explicit StereoWorkspace(ExecutionOptions options = {})
        : options_(options) {}

    const ExecutionOptions& options() const { return options_; }
    void set_options(const ExecutionOptions& options) { options_ = options; }

    PipelineBuffers& buffers() { return buffers_; }
    const PipelineBuffers& buffers() const { return buffers_; }

    size_t allocated_bytes() const {
        return buffers_.allocated_bytes();
    }

    size_t frame_allocation_count() const { return frame_allocation_count_; }
    size_t frame_reuse_count() const { return frame_reuse_count_; }

    // Backward compatibility aliases
    size_t allocation_count() const { return frame_allocation_count_; }
    size_t reuse_count() const { return frame_reuse_count_; }

    void record_frame(int w, int h, int d_max, bool was_reuse) {
        if (was_reuse) {
            ++frame_reuse_count_;
        } else {
            ++frame_allocation_count_;
        }
        last_w_ = w;
        last_h_ = h;
        last_dmax_ = d_max;
    }

    bool can_reuse(int w, int h, int d_max) const {
        return (options_.memory_policy == MemoryPolicy::BoundedReuse) &&
               (w == last_w_) && (h == last_h_) && (d_max == last_dmax_) &&
               (allocated_bytes() > 0);
    }

    void release() {
        buffers_.release();
        last_w_ = 0;
        last_h_ = 0;
        last_dmax_ = 0;
    }

    void reset_counters() {
        frame_allocation_count_ = 0;
        frame_reuse_count_ = 0;
    }

    void trim(size_t max_bytes) {
        if (allocated_bytes() > max_bytes) {
            release();
        }
    }

private:
    ExecutionOptions options_;
    PipelineBuffers buffers_;
    int last_w_ = 0;
    int last_h_ = 0;
    int last_dmax_ = 0;
    size_t frame_allocation_count_ = 0;
    size_t frame_reuse_count_ = 0;
};

} // namespace apg
