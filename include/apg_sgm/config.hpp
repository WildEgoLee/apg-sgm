#pragma once

#include "apg_sgm/types.hpp"

namespace apg {

struct CostParams {
    CensusType census = CensusType::SymmetricCensus9x7;
    float lambda_census = 16.f;
    float lambda_ad = 30.f;
    float lambda_grad = 20.f;
    float eta_ad = 0.4f;
    float mu_grad = 0.3f;
    bool use_ad = true;
    bool use_grad = true;
    int cost_max = 255;
};

struct AggregationParams {
    bool enable = true;
    int max_arm_length = 17;
    int color_threshold = 12;
    int iterations = 1;
};

struct SgmParams {
    PathType paths = PathType::Path4;
    int P1 = 10;
    int P2_base = 120;
    bool adaptive_penalties = true;
    int grad_t1 = 8;
    int grad_t2 = 24;
    float p2_alpha = 0.15f;
    bool piecewise_p2 = true;
};

struct PriorParams {
    bool enable = false;
    float uniqueness_ratio = 0.4f;
    int texture_threshold = 4;
    int lr_max_diff = 1;
    int search_radius = 16;
    int min_supports = 32;
    int max_supports = 4000;
};

struct ConfidenceParams {
    float uniqueness_ratio = 0.15f;
    int lr_max_diff = 1;
    float min_texture = 2.f;
    float reliable_threshold = 0.25f;
};

struct RefineParams {
    bool enable = false;
    int iterations = 3;
    int random_radius = 4;
};

struct PostParams {
    bool subpixel = true;
    bool lr_check = true;
    int lr_max_diff = 1;
    int median_radius = 1;
    int max_fill_gap = 32;
};

enum class HierarchicalFallback {
    NearestValid,
    FullRange,
    ExpandedRadius
};

struct HierarchicalConfig {
    bool enable = false;
    int radius = 4;
    int scale = 2;
    HierarchicalFallback fallback = HierarchicalFallback::NearestValid;
    bool coarse_cross_agg = true;
    bool coarse_fill_holes = true;
    bool adaptive_radius = false;
    float confidence_threshold = 0.5f;
    int expanded_radius = 8;
};

struct PipelineConfig {
    QualityMode mode = QualityMode::Balanced;
    VolumeBackend volume_backend = VolumeBackend::Auto;
    int min_disparity = 0;
    int max_disparity = 128;
    int num_threads = 0;
    // Deprecated compatibility switch. Only honored when volume_backend == Auto.
    // Prefer volume_backend = VolumeBackend::{Auto, Dense, Packed}.
    bool use_packed_volume = false;

    bool resolves_to_packed() const {
        if (volume_backend == VolumeBackend::Packed) return true;
        if (volume_backend == VolumeBackend::Dense) return false;
        return prior.enable || hierarchical.enable || use_packed_volume;
    }

    CostParams cost;
    AggregationParams aggregation;
    SgmParams sgm;
    PriorParams prior;
    ConfidenceParams confidence;
    RefineParams refine;
    PostParams post;
    HierarchicalConfig hierarchical;

    static PipelineConfig from_mode(QualityMode mode, int max_disparity = 128);
    static PipelineConfig quality_v2(int max_disparity = 128);
};

} // namespace apg
