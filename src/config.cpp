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
    }
    return c;
}

} // namespace apg
