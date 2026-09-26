#include "apg_sgm/refiner.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "discrete_cost_lut.hpp"

#include <cmath>
#include <random>

namespace apg {

namespace {

float local_cost(const PipelineConfig& cfg, const PipelineBuffers& buf, const detail::DiscreteCostLut& lut, int x, int y, int d) {
    const int w = buf.left_gray.width();
    const int xr = x - d;
    const int lo = buf.range.dmin.at(x, y);
    const int hi = buf.range.dmax.at(x, y);
    if (xr < 0 || xr >= w || d < lo || d >= hi) return 1e6f;
    const bool sym = cfg.cost.census == CensusType::SymmetricCensus9x7;
    const int pop = sym ? CostComputer::popcount32(buf.census_left[y * w + x] ^ buf.census_right[y * w + xr])
                        : CostComputer::popcount64(buf.census_left64[y * w + x] ^ buf.census_right64[y * w + xr]);
    float c = lut.census[pop];
    if (cfg.cost.use_ad) {
        const int ad = std::abs(static_cast<int>(buf.left_gray.at(x, y)) -
                                static_cast<int>(buf.right_gray.at(xr, y)));
        c += lut.ad[ad];
    }
    if (cfg.cost.use_grad) {
        const int g =
            std::abs(static_cast<int>(buf.left_gx.at(x, y)) - static_cast<int>(buf.right_gx.at(xr, y))) +
            std::abs(static_cast<int>(buf.left_gy.at(x, y)) - static_cast<int>(buf.right_gy.at(xr, y)));
        c += lut.grad[g];
    }
    return c;
}

} // namespace

void Refiner::refine(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    if (!cfg.refine.enable) return;
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    std::mt19937 rng(12345);

    const detail::DiscreteCostLut lut(cfg);

    Image32f work = buf.disparity;
    for (int it = 0; it < cfg.refine.iterations; ++it) {
        const bool fwd = (it % 2) == 0;
        const int radius = std::max(1, cfg.refine.random_radius >> it);
        std::uniform_int_distribution<int> dist(-radius, radius);

        const int y0 = fwd ? 0 : h - 1;
        const int y1 = fwd ? h : -1;
        const int ys = fwd ? 1 : -1;
        const int x0 = fwd ? 0 : w - 1;
        const int x1 = fwd ? w : -1;
        const int xs = fwd ? 1 : -1;
        for (int y = y0; y != y1; y += ys) {
            for (int x = x0; x != x1; x += xs) {
                if (buf.reliable_mask.at(x, y)) continue;
                const int lo = buf.range.dmin.at(x, y);
                const int hi = buf.range.dmax.at(x, y);
                if (hi <= lo) continue;

                const float cur = work.at(x, y);
                int candidates[6];
                int n = 0;
                auto push = [&](float d) {
                    if (d < 0.f) return;
                    candidates[n++] = clampi(static_cast<int>(std::round(d)), lo, hi - 1);
                };
                push(cur);
                if (fwd) {
                    if (x > 0) push(work.at(x - 1, y));
                    if (y > 0) push(work.at(x, y - 1));
                } else {
                    if (x + 1 < w) push(work.at(x + 1, y));
                    if (y + 1 < h) push(work.at(x, y + 1));
                }
                if (cur >= 0.f) push(cur + static_cast<float>(dist(rng)));

                float best_c = 1e9f;
                int best_d = (cur >= 0.f) ? clampi(static_cast<int>(std::round(cur)), lo, hi - 1) : lo;
                for (int i = 0; i < n; ++i) {
                    const float c = local_cost(cfg, buf, lut, x, y, candidates[i]);
                    if (c < best_c) {
                        best_c = c;
                        best_d = candidates[i];
                    }
                }
                work.at(x, y) = static_cast<float>(best_d);
            }
        }
    }
    buf.disparity = std::move(work);
}

} // namespace apg
