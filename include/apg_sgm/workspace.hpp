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

    void release() {
        buffers_.release();
    }

    void trim(size_t max_bytes) {
        if (allocated_bytes() > max_bytes) {
            release();
        }
    }

private:
    ExecutionOptions options_;
    PipelineBuffers buffers_;
};

} // namespace apg
