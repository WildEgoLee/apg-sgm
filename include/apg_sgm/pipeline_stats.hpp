#pragma once

#include <cstddef>

namespace apg {

struct StageTiming {
    double aux_ms = 0.0;
    double prior_ms = 0.0;
    double cost_ms = 0.0;
    double cross_ms = 0.0;
    double right_wta_ms = 0.0;
    double sgm_ms = 0.0;
    double wta_ms = 0.0;
    double confidence_ms = 0.0;
    double refine_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineStats {
    StageTiming timing;

    size_t cost_bytes = 0;
    size_t aggregated_cost_bytes = 0;
    size_t estimated_peak_bytes = 0;

    size_t prior_support_count = 0;
    double mean_search_width = 0.0;
    double search_reduction_ratio = 0.0;

    size_t reliable_pixels = 0;
    size_t unreliable_pixels = 0;
    size_t refine_changed_pixels = 0;
    size_t lr_fail_pixels = 0;

    float reliable_ratio = 0.0f;
    float unreliable_ratio = 0.0f;
    float refine_changed_ratio = 0.0f;
    float lr_fail_ratio = 0.0f;
};

} // namespace apg
