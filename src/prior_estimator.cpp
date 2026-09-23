#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/cost_computer.hpp"

#include <algorithm>
#include <cmath>

namespace apg {

std::vector<SupportMatch> PriorEstimator::extract_supports(const PipelineConfig& cfg,
                                                           const PipelineBuffers& buf) const {
    std::vector<SupportMatch> out;
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    const int d0 = cfg.min_disparity;
    const int d1 = cfg.max_disparity;
    const bool sym = cfg.cost.census == CensusType::SymmetricCensus9x7;
    const int step = 2;
    out.reserve(static_cast<size_t>(w * h) / (step * step * 8));

    for (int y = 2; y < h - 2; y += step) {
        for (int x = 2; x < w - 2; x += step) {
            const int tex = static_cast<int>(buf.left_gx.at(x, y)) + static_cast<int>(buf.left_gy.at(x, y));
            if (tex < cfg.prior.texture_threshold) continue;

            int best_d = d0;
            int best = 255, second = 255;
            for (int d = d0; d < d1; ++d) {
                const int xr = x - d;
                if (xr < 0 || xr >= w) continue;
                int c = 0;
                if (sym) {
                    c = CostComputer::popcount32(buf.census_left[y * w + x] ^
                                                 buf.census_right[y * w + xr]);
                } else {
                    c = CostComputer::popcount64(buf.census_left64[y * w + x] ^
                                                 buf.census_right64[y * w + xr]);
                }
                if (c < best) {
                    second = best;
                    best = c;
                    best_d = d;
                } else if (c < second) {
                    second = c;
                }
            }
            if (second <= 0) continue;
            const float uniq = static_cast<float>(second - best) / static_cast<float>(second);
            if (uniq < cfg.prior.uniqueness_ratio) continue;

            const int xr = x - best_d;
            if (xr < 1 || xr >= w - 1) continue;
            int rbest = 255, rbest_d = 0;
            for (int d = d0; d < d1; ++d) {
                const int xl = xr + d;
                if (xl < 0 || xl >= w) continue;
                int c = 0;
                if (sym) {
                    c = CostComputer::popcount32(buf.census_left[y * w + xl] ^
                                                 buf.census_right[y * w + xr]);
                } else {
                    c = CostComputer::popcount64(buf.census_left64[y * w + xl] ^
                                                 buf.census_right64[y * w + xr]);
                }
                if (c < rbest) {
                    rbest = c;
                    rbest_d = d;
                }
            }
            if (std::abs(rbest_d - best_d) > cfg.prior.lr_max_diff) continue;

            SupportMatch m;
            m.x = x;
            m.y = y;
            m.disparity = static_cast<float>(best_d);
            m.confidence = uniq;
            out.push_back(m);
            if (static_cast<int>(out.size()) >= cfg.prior.max_supports) return out;
        }
    }
    return out;
}

void PriorEstimator::interpolate_prior(const std::vector<SupportMatch>& supports,
                                       PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    buf.d_prior = Image32f(w, h, -1.f);
    if (supports.empty()) return;

    constexpr int cell = 16;
    const int gw = (w + cell - 1) / cell;
    const int gh = (h + cell - 1) / cell;
    std::vector<float> acc(static_cast<size_t>(gw) * gh, 0.f);
    std::vector<float> wgt(static_cast<size_t>(gw) * gh, 0.f);
    for (const auto& s : supports) {
        const int gx = clampi(s.x / cell, 0, gw - 1);
        const int gy = clampi(s.y / cell, 0, gh - 1);
        const int i = gy * gw + gx;
        acc[i] += s.disparity * s.confidence;
        wgt[i] += s.confidence;
    }
    std::vector<float> grid(static_cast<size_t>(gw) * gh, -1.f);
    for (int i = 0; i < gw * gh; ++i) {
        if (wgt[i] > 1e-6f) grid[i] = acc[i] / wgt[i];
    }
    auto fill_grid = [&]() {
        std::vector<float> nxt = grid;
        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                if (grid[gy * gw + gx] >= 0.f) continue;
                float a = 0.f, ww = 0.f;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int nx = gx + ox, ny = gy + oy;
                        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
                        const float v = grid[ny * gw + nx];
                        if (v < 0.f) continue;
                        a += v;
                        ww += 1.f;
                    }
                }
                if (ww > 0.f) nxt[gy * gw + gx] = a / ww;
            }
        }
        grid.swap(nxt);
    };
    for (int i = 0; i < 8; ++i) fill_grid();

    for (int y = 0; y < h; ++y) {
        const float fy = (static_cast<float>(y) + 0.5f) / cell - 0.5f;
        const int gy = clampi(static_cast<int>(std::floor(fy)), 0, gh - 1);
        const int gy1 = std::min(gy + 1, gh - 1);
        const float ty = clampf(fy - static_cast<float>(gy), 0.f, 1.f);
        for (int x = 0; x < w; ++x) {
            const float fx = (static_cast<float>(x) + 0.5f) / cell - 0.5f;
            const int gx = clampi(static_cast<int>(std::floor(fx)), 0, gw - 1);
            const int gx1 = std::min(gx + 1, gw - 1);
            const float tx = clampf(fx - static_cast<float>(gx), 0.f, 1.f);
            const float v00 = grid[gy * gw + gx];
            const float v10 = grid[gy * gw + gx1];
            const float v01 = grid[gy1 * gw + gx];
            const float v11 = grid[gy1 * gw + gx1];
            auto pick = [](float v) { return v < 0.f ? 0.f : v; };
            auto wgtv = [](float v) { return v < 0.f ? 0.f : 1.f; };
            const float a =
                (1 - tx) * (1 - ty) * pick(v00) + tx * (1 - ty) * pick(v10) +
                (1 - tx) * ty * pick(v01) + tx * ty * pick(v11);
            const float ww =
                (1 - tx) * (1 - ty) * wgtv(v00) + tx * (1 - ty) * wgtv(v10) +
                (1 - tx) * ty * wgtv(v01) + tx * ty * wgtv(v11);
            buf.d_prior.at(x, y) = (ww > 1e-6f) ? a / ww : -1.f;
        }
    }
}

void PriorEstimator::apply_search_range(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    buf.range.allocate(w, h, cfg.min_disparity, cfg.max_disparity);
    if (!cfg.prior.enable || buf.d_prior.empty()) return;
    const int R = cfg.prior.search_radius;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dp = buf.d_prior.at(x, y);
            if (dp < 0.f) continue;
            const int lo = static_cast<int>(std::floor(dp)) - R;
            const int hi = static_cast<int>(std::ceil(dp)) + R + 1;
            buf.range.set_pixel(x, y, lo, hi);
        }
    }
}

void PriorEstimator::estimate(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    buf.range.allocate(w, h, cfg.min_disparity, cfg.max_disparity);
    buf.d_prior = Image32f(w, h, -1.f);
    if (!cfg.prior.enable) return;

    auto supports = extract_supports(cfg, buf);
    if (static_cast<int>(supports.size()) < cfg.prior.min_supports) return;
    interpolate_prior(supports, buf);
    apply_search_range(cfg, buf);
}

} // namespace apg
