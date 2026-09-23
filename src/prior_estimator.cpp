#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/cost_computer.hpp"

#include <algorithm>
#include <cmath>

namespace apg {

std::vector<SupportMatch> PriorEstimator::extract_supports(const PipelineConfig& cfg,
                                                           const PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    const int d0 = cfg.min_disparity;
    const int d1 = cfg.max_disparity;
    const bool sym = cfg.cost.census == CensusType::SymmetricCensus9x7;
    const int step = 2;

    constexpr int cell = 16;
    const int gw = (w + cell - 1) / cell;
    const int gh = (h + cell - 1) / cell;
    const int n_cells = gw * gh;
    const int hard_max = cfg.prior.max_supports;

    // Per-cell candidate collections
    std::vector<std::vector<SupportMatch>> cell_candidates(n_cells);
    const int max_cand_per_cell = std::max(8, (hard_max * 2 + n_cells - 1) / std::max(1, n_cells));

    for (int y = 2; y < h - 2; y += step) {
        const int gy = clampi(y / cell, 0, gh - 1);
        for (int x = 2; x < w - 2; x += step) {
            const int gx = clampi(x / cell, 0, gw - 1);
            const int ci = gy * gw + gx;
            if (static_cast<int>(cell_candidates[ci].size()) >= max_cand_per_cell) continue;

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
            cell_candidates[ci].push_back(m);
        }
    }

    // Stratified budget allocation:
    // 1) Spatially uniform distribution across all cells (no top/bottom starvation)
    // 2) out.size() <= hard_max (strictly honoring max_supports)
    std::vector<int> nonempty_cells;
    nonempty_cells.reserve(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        if (!cell_candidates[i].empty()) {
            nonempty_cells.push_back(i);
        }
    }

    std::vector<SupportMatch> out;
    out.reserve(std::min(static_cast<size_t>(hard_max), static_cast<size_t>(n_cells * 4)));

    if (static_cast<int>(nonempty_cells.size()) <= hard_max) {
        // Case A: Budget is sufficient to cover all nonempty cells.
        // Each nonempty cell gets at least base_quota supports, remainder is distributed round-robin.
        std::vector<int> taken(n_cells, 0);

        // Pass 1: Base quota for each nonempty cell
        const int base_quota = std::max(1, hard_max / std::max(1, static_cast<int>(nonempty_cells.size())));
        for (int ci : nonempty_cells) {
            const int take = std::min(base_quota, static_cast<int>(cell_candidates[ci].size()));
            for (int k = 0; k < take && static_cast<int>(out.size()) < hard_max; ++k) {
                out.push_back(cell_candidates[ci][k]);
                taken[ci]++;
            }
        }

        // Pass 2: Redistribution of remaining budget in round-robin fashion across cells that have remaining candidates
        bool added = true;
        while (added && static_cast<int>(out.size()) < hard_max) {
            added = false;
            for (int ci : nonempty_cells) {
                if (static_cast<int>(out.size()) >= hard_max) break;
                if (taken[ci] < static_cast<int>(cell_candidates[ci].size())) {
                    out.push_back(cell_candidates[ci][taken[ci]]);
                    taken[ci]++;
                    added = true;
                }
            }
        }
    } else {
        // Case B: High-resolution / large grid where nonempty_cells > hard_max.
        // Deterministic uniform spatial subsampling across the entire image extent.
        const size_t num_nonempty = nonempty_cells.size();
        for (int k = 0; k < hard_max; ++k) {
            const size_t sample_idx = (static_cast<uint64_t>(k) * num_nonempty) / hard_max;
            const int ci = nonempty_cells[sample_idx];
            if (!cell_candidates[ci].empty()) {
                out.push_back(cell_candidates[ci].front());
            }
        }
    }

    return out;
}

void PriorEstimator::interpolate_prior(const std::vector<SupportMatch>& supports,
                                       PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    buf.d_prior = Image32f(w, h, -1.f);
    buf.prior_confidence = Image32f(w, h, 0.f);
    buf.prior_spread = Image32f(w, h, 999.f);
    buf.d_prior_min = Image32f(w, h, -1.f);
    buf.d_prior_max = Image32f(w, h, -1.f);
    if (supports.empty()) return;

    constexpr int cell = 16;
    const int gw = (w + cell - 1) / cell;
    const int gh = (h + cell - 1) / cell;
    const int n_cells = gw * gh;

    std::vector<std::vector<std::pair<float, float>>> cell_supports(n_cells);
    for (const auto& s : supports) {
        const int gx = clampi(s.x / cell, 0, gw - 1);
        const int gy = clampi(s.y / cell, 0, gh - 1);
        cell_supports[gy * gw + gx].push_back({s.disparity, s.confidence});
    }

    std::vector<float> grid_disp(n_cells, -1.f);
    std::vector<float> grid_conf(n_cells, 0.f);
    std::vector<float> grid_spread(n_cells, -1.f);
    std::vector<float> grid_dmin(n_cells, -1.f);
    std::vector<float> grid_dmax(n_cells, -1.f);

    for (int i = 0; i < n_cells; ++i) {
        auto& pts = cell_supports[i];
        if (pts.empty()) continue;

        std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        grid_dmin[i] = pts.front().first;
        grid_dmax[i] = pts.back().first;
        const float intra_span = grid_dmax[i] - grid_dmin[i];

        float total_w = 0.f;
        for (const auto& p : pts) total_w += p.second;
        if (total_w <= 1e-6f) continue;

        // Weighted median disparity
        float half_w = total_w * 0.5f;
        float cur_w = 0.f;
        float d_med = pts.front().first;
        for (const auto& p : pts) {
            cur_w += p.second;
            if (cur_w >= half_w) {
                d_med = p.first;
                break;
            }
        }
        grid_disp[i] = d_med;

        // MAD spread
        std::vector<std::pair<float, float>> devs;
        devs.reserve(pts.size());
        for (const auto& p : pts) {
            devs.push_back({std::abs(p.first - d_med), p.second});
        }
        std::sort(devs.begin(), devs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        cur_w = 0.f;
        float dev_med = devs.front().first;
        for (const auto& d : devs) {
            cur_w += d.second;
            if (cur_w >= half_w) {
                dev_med = d.first;
                break;
            }
        }
        grid_spread[i] = std::max(dev_med * 1.4826f, intra_span * 0.5f);

        const float mean_conf = total_w / static_cast<float>(pts.size());
        const float count_factor = clampf(static_cast<float>(pts.size()) / 3.f, 0.3f, 1.f);
        grid_conf[i] = clampf(mean_conf * count_factor, 0.f, 1.f);
    }

    auto fill_grids = [&]() -> bool {
        std::vector<float> nxt_d = grid_disp;
        std::vector<float> nxt_c = grid_conf;
        std::vector<float> nxt_s = grid_spread;
        std::vector<float> nxt_min = grid_dmin;
        std::vector<float> nxt_max = grid_dmax;
        bool changed = false;
        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                const int i = gy * gw + gx;
                if (grid_disp[i] >= 0.f) continue;
                float ad = 0.f, ac = 0.f, as = 0.f, ww = 0.f;
                float cur_min = 1e9f, cur_max = -1e9f;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int nx = gx + ox, ny = gy + oy;
                        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
                        const int ni = ny * gw + nx;
                        const float v = grid_disp[ni];
                        if (v < 0.f) continue;
                        ad += v;
                        ac += grid_conf[ni];
                        as += (grid_spread[ni] >= 0.f ? grid_spread[ni] : 8.f);
                        cur_min = std::min(cur_min, grid_dmin[ni] >= 0.f ? grid_dmin[ni] : v);
                        cur_max = std::max(cur_max, grid_dmax[ni] >= 0.f ? grid_dmax[ni] : v);
                        ww += 1.f;
                    }
                }
                if (ww > 0.f) {
                    nxt_d[i] = ad / ww;
                    nxt_c[i] = (ac / ww) * 0.85f; // decay confidence for hole filled cells
                    nxt_s[i] = as / ww;
                    nxt_min[i] = cur_min;
                    nxt_max[i] = cur_max;
                    changed = true;
                }
            }
        }
        grid_disp.swap(nxt_d);
        grid_conf.swap(nxt_c);
        grid_spread.swap(nxt_s);
        grid_dmin.swap(nxt_min);
        grid_dmax.swap(nxt_max);
        return changed;
    };
    const int max_iters = std::max(gw, gh);
    for (int iter = 0; iter < max_iters; ++iter) {
        if (!fill_grids()) break;
    }

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

            const int i00 = gy * gw + gx;
            const int i10 = gy * gw + gx1;
            const int i01 = gy1 * gw + gx;
            const int i11 = gy1 * gw + gx1;

            auto pick = [](float v) { return v < 0.f ? 0.f : v; };
            auto wgtv = [](float v) { return v < 0.f ? 0.f : 1.f; };

            const float w00 = (1.f - tx) * (1.f - ty) * wgtv(grid_disp[i00]);
            const float w10 = tx * (1.f - ty) * wgtv(grid_disp[i10]);
            const float w01 = (1.f - tx) * ty * wgtv(grid_disp[i01]);
            const float w11 = tx * ty * wgtv(grid_disp[i11]);
            const float ww = w00 + w10 + w01 + w11;

            if (ww > 1e-6f) {
                const float dp =
                    (w00 * pick(grid_disp[i00]) + w10 * pick(grid_disp[i10]) +
                     w01 * pick(grid_disp[i01]) + w11 * pick(grid_disp[i11])) / ww;
                buf.d_prior.at(x, y) = dp;
                buf.prior_confidence.at(x, y) =
                    (w00 * grid_conf[i00] + w10 * grid_conf[i10] +
                     w01 * grid_conf[i01] + w11 * grid_conf[i11]) / ww;

                float cmin = 1e9f, cmax = -1e9f;
                for (int idx : {i00, i10, i01, i11}) {
                    if (grid_disp[idx] >= 0.f) {
                        const float mn = grid_dmin[idx] >= 0.f ? grid_dmin[idx] : grid_disp[idx];
                        const float mx = grid_dmax[idx] >= 0.f ? grid_dmax[idx] : grid_disp[idx];
                        cmin = std::min(cmin, mn);
                        cmax = std::max(cmax, mx);
                    }
                }
                const float local_spread =
                    (w00 * pick(grid_spread[i00]) + w10 * pick(grid_spread[i10]) +
                     w01 * pick(grid_spread[i01]) + w11 * pick(grid_spread[i11])) / ww;
                buf.prior_spread.at(x, y) = local_spread;

                buf.d_prior_min.at(x, y) = (cmin <= cmax) ? std::min(cmin, dp) : dp;
                buf.d_prior_max.at(x, y) = (cmin <= cmax) ? std::max(cmax, dp) : dp;
            } else {
                buf.d_prior.at(x, y) = -1.f;
                buf.prior_confidence.at(x, y) = 0.f;
                buf.prior_spread.at(x, y) = 999.f;
                buf.d_prior_min.at(x, y) = -1.f;
                buf.d_prior_max.at(x, y) = -1.f;
            }
        }
    }
}

void PriorEstimator::apply_search_range(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    if (!cfg.prior.enable || buf.d_prior.empty()) return;
    const int default_R = cfg.prior.search_radius;
    const int global_D = cfg.max_disparity - cfg.min_disparity;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dp = buf.d_prior.at(x, y);
            if (dp < 0.f) continue;

            const float conf = !buf.prior_confidence.empty() ? buf.prior_confidence.at(x, y) : 0.f;
            const float spread = !buf.prior_spread.empty() ? buf.prior_spread.at(x, y) : 999.f;
            const float pmin = (!buf.d_prior_min.empty() && buf.d_prior_min.at(x, y) >= 0.f) ? buf.d_prior_min.at(x, y) : dp;
            const float pmax = (!buf.d_prior_max.empty() && buf.d_prior_max.at(x, y) >= 0.f) ? buf.d_prior_max.at(x, y) : dp;

            int R = default_R;
            if (conf < 0.25f) {
                R = global_D;
            } else if (conf > 0.85f && spread < 1.5f) {
                R = 8;
            } else if (conf > 0.60f && spread < 3.0f) {
                R = static_cast<int>(12.f + 1.5f * spread + 0.5f);
            } else if (conf > 0.40f && spread < 6.0f) {
                R = static_cast<int>(18.f + 2.0f * spread + 0.5f);
            } else {
                // High spread, depth edge or uncertain transition: conservative search
                R = std::max(24, static_cast<int>(16.f + 2.0f * spread + 0.5f));
            }
            R = std::min(R, global_D);

            constexpr int kEnvelopeMargin = 4;
            const int bound_lo = static_cast<int>(std::floor(pmin)) - kEnvelopeMargin;
            const int bound_hi = static_cast<int>(std::ceil(pmax)) + kEnvelopeMargin + 1;

            const int lo = std::min(static_cast<int>(std::floor(dp)) - R, bound_lo);
            int hi = std::max(static_cast<int>(std::ceil(dp)) + R + 1, bound_hi);
            hi = std::min(hi, x + 1);
            buf.range.set_pixel(x, y, lo, hi);
        }
    }
}

void PriorEstimator::estimate(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    buf.range.allocate(w, h, cfg.min_disparity, cfg.max_disparity);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = cfg.min_disparity;
            const int hi = std::min(cfg.max_disparity, x + 1);
            buf.range.set_pixel(x, y, lo, hi);
        }
    }
    buf.d_prior = Image32f(w, h, -1.f);
    buf.supports.clear();
    buf.support_count = 0;
    if (!cfg.prior.enable) return;

    buf.supports = extract_supports(cfg, buf);
    buf.support_count = buf.supports.size();
    if (static_cast<int>(buf.supports.size()) < cfg.prior.min_supports) return;
    interpolate_prior(buf.supports, buf);
    apply_search_range(cfg, buf);
}

} // namespace apg
