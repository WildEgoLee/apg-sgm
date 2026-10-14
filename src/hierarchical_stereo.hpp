#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

#include <functional>
#include <string>

namespace apg {

enum class HierarchicalFallback {
    FullRange,
    NearestValid,
    ExpandedRadius
};

inline std::string hierarchical_fallback_to_string(HierarchicalFallback f) {
    switch (f) {
        case HierarchicalFallback::FullRange: return "full_range";
        case HierarchicalFallback::NearestValid: return "nearest_valid";
        case HierarchicalFallback::ExpandedRadius: return "expanded_radius";
    }
    return "unknown";
}

inline bool parse_hierarchical_fallback(const std::string& str, HierarchicalFallback& out) {
    if (str == "full_range" || str == "FULL_RANGE" || str == "full") {
        out = HierarchicalFallback::FullRange;
        return true;
    }
    if (str == "nearest_valid" || str == "NEAREST_VALID" || str == "nearest") {
        out = HierarchicalFallback::NearestValid;
        return true;
    }
    if (str == "expanded_radius" || str == "EXPANDED_RADIUS" || str == "expanded") {
        out = HierarchicalFallback::ExpandedRadius;
        return true;
    }
    return false;
}

struct HierarchicalConfig {
    bool enable = false;
    int radius = 8;
    int scale = 2;
    HierarchicalFallback fallback = HierarchicalFallback::NearestValid;
    bool coarse_cross_agg = true;
    bool coarse_fill_holes = true;
};

struct HierarchicalTelemetry {
    double avg_states_per_pixel = 0.0;
    float coarse_valid_ratio = 0.0f;
    float fallback_pixel_ratio = 0.0f;
    float full_range_pixel_ratio = 0.0f;
    double coarse_time_ms = 0.0;
};

class ExperimentalHierarchicalScope {
public:
    explicit ExperimentalHierarchicalScope(const HierarchicalConfig& cfg) {
        prev_cfg_ = current_config();
        current_config() = cfg;
        current_telemetry() = HierarchicalTelemetry{};
    }

    ~ExperimentalHierarchicalScope() {
        current_config() = prev_cfg_;
    }

    static bool is_enabled() {
        return current_config().enable;
    }

    static HierarchicalConfig& current_config() {
        static thread_local HierarchicalConfig s_cfg;
        return s_cfg;
    }

    static HierarchicalTelemetry& current_telemetry() {
        static thread_local HierarchicalTelemetry s_telem;
        return s_telem;
    }

private:
    HierarchicalConfig prev_cfg_;
};

class HierarchicalStereo {
public:
    static bool estimate_search_range(
        const HierarchicalConfig& hcfg,
        const PipelineConfig& cfg,
        PipelineBuffers& buf,
        HierarchicalTelemetry* telemetry = nullptr,
        std::function<void(size_t coarse_allocated_bytes)> on_coarse_sample = nullptr,
        int num_threads = 0
    );
};

} // namespace apg
