#include "apg_sgm/config.hpp"

namespace apg {

PipelineConfig PipelineConfig::from_mode(QualityMode mode, int max_disparity) {
    PipelineConfig c;
    c.mode = mode;
    c.max_disparity = max_disparity;
    c.min_disparity = 0;

    switch (mode) {
    case QualityMode::Fast:
        c.cost.use_ad = false;
        c.cost.use_grad = false;
        c.aggregation.enable = false;
        c.sgm.paths = PathType::Path4;
        c.sgm.adaptive_penalties = false;
        c.prior.enable = false;
        c.refine.enable = false;
        c.post.subpixel = true;
        c.post.lr_check = true;
        break;
    case QualityMode::Balanced:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.paths = PathType::Path4;
        c.sgm.adaptive_penalties = true;
        c.prior.enable = false;
        c.refine.enable = false;
        break;
    case QualityMode::HighQuality:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.sgm.paths = PathType::Path8;
        c.sgm.adaptive_penalties = true;
        c.prior.enable = true;
        c.refine.enable = true;
        c.refine.iterations = 3;
        break;
    case QualityMode::QualityV2:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.paths = PathType::Path4;
        c.sgm.adaptive_penalties = true;
        c.prior.enable = true;
        c.refine.enable = true;
        c.refine.iterations = 3;
        c.hierarchical.enable = true;
        c.hierarchical.radius = 4;
        c.hierarchical.scale = 2;
        c.hierarchical.fallback = HierarchicalFallback::NearestValid;
        c.hierarchical.coarse_cross_agg = true;
        c.hierarchical.coarse_fill_holes = true;
        c.post.subpixel = true;
        c.post.lr_check = true;
        c.post.max_fill_gap = 48;
        c.post.shadow_reclass = true;
        break;
    }
    return c;
}

PipelineConfig PipelineConfig::quality_v2(int max_disparity) {
    return from_mode(QualityMode::QualityV2, max_disparity);
}

} // namespace apg
