#include "apg_sgm/confidence_estimator.hpp"

#include <cmath>

namespace apg {

void ConfidenceEstimator::estimate(const PipelineConfig& cfg, PipelineBuffers& buf,
                                   const Image32f& best_cost, const Image32f& second_cost) const {
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    buf.confidence = Image32f(w, h, 0.f);
    buf.reliable_mask = Image8u1(w, h, 0);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float c1 = best_cost.at(x, y);
            const float c2 = second_cost.at(x, y);
            float uniq = 0.f;
            if (c1 > 1e-3f) uniq = (c2 - c1) / (c1 + 1e-3f);
            else uniq = 1.f;

            float lr = 1.f;
            if (!buf.disparity_right.empty()) {
                const float d = buf.disparity.at(x, y);
                const int xr = static_cast<int>(std::round(static_cast<float>(x) - d));
                if (xr < 0 || xr >= w || d < 0.f) {
                    lr = 0.f;
                } else {
                    const float dr = buf.disparity_right.at(xr, y);
                    if (dr < 0.f || std::abs(d - dr) > static_cast<float>(cfg.confidence.lr_max_diff) + 0.5f) {
                        lr = 0.f;
                    }
                }
            }

            const float tex = static_cast<float>(buf.left_gx.at(x, y) + buf.left_gy.at(x, y));
            const float tex_term = clampf(tex / 16.f, 0.f, 1.f);

            float conf = clampf(uniq, 0.f, 4.f);
            conf = conf * (0.5f + 0.5f * tex_term);
            if (lr < 0.5f) conf *= 0.25f;
            buf.confidence.at(x, y) = conf;

            const bool reliable = conf >= cfg.refine.conf_threshold &&
                                  uniq >= cfg.confidence.uniqueness_ratio &&
                                  tex >= cfg.confidence.min_texture &&
                                  buf.disparity.at(x, y) >= 0.f;
            buf.reliable_mask.at(x, y) = reliable ? 1 : 0;
            if (!reliable) {
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::LowConfidence);
            }
        }
    }
}

} // namespace apg
