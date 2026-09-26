#pragma once

#include "apg_sgm/config.hpp"

#include <algorithm>
#include <cmath>

namespace apg::detail {

struct DiscreteCostLut {
    float census[65]{};
    float ad[256]{};
    float grad[511]{};

    explicit DiscreteCostLut(const PipelineConfig& cfg) {
        const float lc = std::max(cfg.cost.lambda_census, 1e-3f);
        const float la = std::max(cfg.cost.lambda_ad, 1e-3f);
        const float lg = std::max(cfg.cost.lambda_grad, 1e-3f);

        for (int i = 0; i <= 64; ++i) {
            census[i] = 1.f - std::exp(-static_cast<float>(i) / lc);
        }

        if (cfg.cost.use_ad) {
            for (int i = 0; i < 256; ++i) {
                ad[i] = cfg.cost.eta_ad *
                        (1.f - std::exp(-static_cast<float>(i) / la));
            }
        }

        if (cfg.cost.use_grad) {
            for (int i = 0; i < 511; ++i) {
                grad[i] = cfg.cost.mu_grad *
                          (1.f - std::exp(-static_cast<float>(i) / lg));
            }
        }
    }
};

} // namespace apg::detail
