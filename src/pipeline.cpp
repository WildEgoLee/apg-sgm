#include "apg_sgm/pipeline.hpp"

#include "apg_sgm/confidence_estimator.hpp"
#include "apg_sgm/cost_aggregator.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/postprocess.hpp"
#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/refiner.hpp"
#include "apg_sgm/sgm_optimizer.hpp"

#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg {

namespace {

void wta_right_from_left_volume(const PipelineConfig& cfg, const CostVolume& vol,
                                Image32f& disp_right) {
    const int w = vol.width();
    const int h = vol.height();
    const int d0 = vol.d0();
    const int D = vol.D();
    disp_right = Image32f(w, h, -1.f);
    for (int y = 0; y < h; ++y) {
        for (int xr = 0; xr < w; ++xr) {
            int best_d = -1;
            uint32_t best = UINT32_MAX;
            for (int d = d0; d < d0 + D; ++d) {
                const int xl = xr + d;
                if (xl < 0 || xl >= w) continue;
                const uint16_t c = vol.slice(xl, y)[d - d0];
                if (c == kInvalidCost) continue;
                if (c < best) {
                    best = c;
                    best_d = d;
                }
            }
            if (best_d >= 0) {
                float dval = static_cast<float>(best_d);
                if (cfg.post.subpixel) {
                    const int di = best_d - d0;
                    const int xl = xr + best_d;
                    if (xl >= 0 && xl < w && di > 0 && di + 1 < D) {
                        const uint16_t* s = vol.slice(xl, y);
                        if (s[di - 1] != kInvalidCost && s[di + 1] != kInvalidCost) {
                            const float cm = static_cast<float>(s[di - 1]);
                            const float c0 = static_cast<float>(s[di]);
                            const float cp = static_cast<float>(s[di + 1]);
                            const float denom = cm - 2.f * c0 + cp;
                            if (std::abs(denom) > 1e-6f) {
                                float delta = 0.5f * (cm - cp) / denom;
                                delta = clampf(delta, -0.5f, 0.5f);
                                dval += delta;
                            }
                        }
                    }
                }
                disp_right.at(xr, y) = dval;
            }
        }
    }
}

} // namespace

StereoMatcher::StereoMatcher(PipelineConfig cfg) : cfg_(std::move(cfg)) {}

bool StereoMatcher::compute(const Image8& left, const Image8& right, PipelineBuffers& out) const {
    last_error_.clear();
    if (left.empty() || right.empty()) {
        last_error_ = "empty input";
        return false;
    }
    if (left.width() != right.width() || left.height() != right.height()) {
        last_error_ = "left/right size mismatch (images must be rectified and aligned)";
        return false;
    }
    if (cfg_.max_disparity <= cfg_.min_disparity) {
        last_error_ = "invalid disparity range";
        return false;
    }

#if defined(_OPENMP)
    if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
#endif

    CostComputer cost;
    PriorEstimator prior;
    CostAggregator agg;
    SgmOptimizer sgm;
    ConfidenceEstimator conf;
    Refiner refiner;
    PostProcessor post;

    cost.compute_aux(left, right, cfg_, out);
    prior.estimate(cfg_, out);
    cost.compute_volume(cfg_, out);
    agg.aggregate(cfg_, out);

    // Compute right disparity from preconditioned cost volume before left SGM
    wta_right_from_left_volume(cfg_, out.cost, out.disparity_right);

    // Optimize left volume with SGM streaming into out.cost32
    sgm.optimize(cfg_, out);

    Image32f best, second;
    sgm.winner_take_all(cfg_, out.cost32, out.range, out.disparity, &best, &second);

    conf.estimate(cfg_, out, best, second);
    refiner.refine(cfg_, out);
    post.left_right_check(cfg_, out);
    post.fill_holes(cfg_, out);
    post.median(cfg_, out);
    return true;
}

} // namespace apg
