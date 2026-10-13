#pragma once

#include <cstddef>

namespace apg {

enum class MemoryPolicy {
    ReleasePerFrame = 0,
    BoundedReuse = 1
};

struct ExecutionOptions {
    MemoryPolicy memory_policy = MemoryPolicy::ReleasePerFrame;
    size_t max_retained_bytes = 0; // 0 = unconstrained if BoundedReuse, or strict limit
    int num_threads = 0;           // 0 = inherit from PipelineConfig / OpenMP
};

} // namespace apg
