#include "hierarchical_stereo.hpp"

#include "apg_sgm/confidence_estimator.hpp"
#include "apg_sgm/cost_aggregator.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/postprocess.hpp"
#include "apg_sgm/sgm_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg {

namespace {

inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

void wta_right_from_packed(const PipelineConfig& cfg, const PackedCostVolume16& vol,
                           Image32f& disp_right) {
    const int w = vol.width();
    const int h = vol.height();
    disp_right = Image32f(w, h, -1.f);
    if (vol.empty() || w <= 0 || h <= 0) return;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<uint16_t> best_cost(w);
        std::vector<int> best_d(w);
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            std::fill(best_cost.begin(), best_cost.end(), kInvalidCost);
            std::fill(best_d.begin(), best_d.end(), -1);

            for (int xl = 0; xl < w; ++xl) {
                const int lo = vol.dmin(xl, y);
                const int hi = vol.dmax(xl, y);
                const int count = hi - lo;
                if (count <= 0) continue;

                const uint16_t* s = vol.slice(xl, y);
                for (int di = 0; di < count; ++di) {
                    const int d = lo + di;
                    const int xr = xl - d;
                    if (xr < 0 || xr >= w) continue;

                    const uint16_t c = s[di];
                    if (c == kInvalidCost) continue;

                    if (c < best_cost[xr]) {
                        best_cost[xr] = c;
                        best_d[xr] = d;
                    }
                }
            }

            for (int xr = 0; xr < w; ++xr) {
                const int bd = best_d[xr];
                if (bd >= 0) {
                    float dval = static_cast<float>(bd);
                    if (cfg.post.subpixel) {
                        const int xl_m = xr + bd - 1;
                        const int xl_0 = xr + bd;
                        const int xl_p = xr + bd + 1;
                        if (xl_m >= 0 && xl_m < w &&
                            xl_0 >= 0 && xl_0 < w &&
                            xl_p >= 0 && xl_p < w &&
                            vol.contains(xl_m, y, bd - 1) &&
                            vol.contains(xl_0, y, bd) &&
                            vol.contains(xl_p, y, bd + 1)) {
                            const uint16_t cm = vol.at(xl_m, y, bd - 1);
                            const uint16_t c0 = vol.at(xl_0, y, bd);
                            const uint16_t cp = vol.at(xl_p, y, bd + 1);
                            if (cm != kInvalidCost && c0 != kInvalidCost && cp != kInvalidCost) {
                                const float fcm = static_cast<float>(cm);
                                const float fc0 = static_cast<float>(c0);
                                const float fcp = static_cast<float>(cp);
                                const float denom = fcm - 2.f * fc0 + fcp;
                                if (std::abs(denom) > 1e-6f) {
                                    float delta = 0.5f * (fcm - fcp) / denom;
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
}

} // namespace

bool HierarchicalStereo::estimate_search_range(
    const HierarchicalConfig& hcfg,
    const PipelineConfig& cfg,
    PipelineBuffers& buf,
    HierarchicalTelemetry* telemetry,
    std::function<void()> on_coarse_alloc,
    int num_threads
) {
    const int W = buf.left_gray.width();
    const int H = buf.left_gray.height();
    if (W < 32 || H < 32) {
        // Fallback to standard full range for tiny dimensions
        buf.range.allocate(W, H, cfg.min_disparity, cfg.max_disparity);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                buf.range.set_pixel(x, y, cfg.min_disparity, std::min(cfg.max_disparity, x + 1));
            }
        }
        return false;
    }

    const auto t_start = std::chrono::steady_clock::now();

    const int W_c = W / 2;
    const int H_c = H / 2;

    if (!buf.coarse_buffers) {
        buf.coarse_buffers = std::make_shared<PipelineBuffers>();
    }
    PipelineBuffers& cbuf = *buf.coarse_buffers;

    // 1. 2x box downsampling of grayscale images
    cbuf.left = Image8(W_c, H_c, 1);
    cbuf.right = Image8(W_c, H_c, 1);

#if defined(_OPENMP)
    if (num_threads > 0) omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(static)
#endif
    for (int yc = 0; yc < H_c; ++yc) {
        const int y0 = yc * 2;
        const int y1 = std::min(y0 + 1, H - 1);
        for (int xc = 0; xc < W_c; ++xc) {
            const int x0 = xc * 2;
            const int x1 = std::min(x0 + 1, W - 1);

            const int sum_l = static_cast<int>(buf.left_gray.at(x0, y0)) +
                              static_cast<int>(buf.left_gray.at(x1, y0)) +
                              static_cast<int>(buf.left_gray.at(x0, y1)) +
                              static_cast<int>(buf.left_gray.at(x1, y1));
            cbuf.left.at(xc, yc) = static_cast<uint8_t>((sum_l + 2) / 4);

            const int sum_r = static_cast<int>(buf.right_gray.at(x0, y0)) +
                              static_cast<int>(buf.right_gray.at(x1, y0)) +
                              static_cast<int>(buf.right_gray.at(x0, y1)) +
                              static_cast<int>(buf.right_gray.at(x1, y1));
            cbuf.right.at(xc, yc) = static_cast<uint8_t>((sum_r + 2) / 4);
        }
    }

    // 2. Coarse configuration
    PipelineConfig cfg_c = cfg;
    cfg_c.min_disparity = std::max(0, cfg.min_disparity / 2);
    cfg_c.max_disparity = std::max(cfg_c.min_disparity + 1, (cfg.max_disparity + 1) / 2);
    cfg_c.prior.enable = false;
    cfg_c.refine.enable = false;
    cfg_c.aggregation.enable = hcfg.coarse_cross_agg;
    cfg_c.volume_backend = VolumeBackend::Packed;
    cfg_c.post.subpixel = true;
    cfg_c.post.lr_check = true;
    cfg_c.post.median_radius = 1;
    cfg_c.post.max_fill_gap = hcfg.coarse_fill_holes ? 16 : 0;
    if (num_threads > 0) cfg_c.num_threads = num_threads;

    // 3. Run coarse matching
    CostComputer cost_c;
    cost_c.compute_aux(ImageView8(cbuf.left), ImageView8(cbuf.right), cfg_c, cbuf);

    cbuf.range.allocate(W_c, H_c, cfg_c.min_disparity, cfg_c.max_disparity);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H_c; ++y) {
        for (int x = 0; x < W_c; ++x) {
            cbuf.range.set_pixel(x, y, cfg_c.min_disparity, std::min(cfg_c.max_disparity, x + 1));
        }
    }

    auto layout = PackedVolumeLayout::from_range(cbuf.range);
    cbuf.packed_cost.allocate_for_overwrite(layout);

    // Capture memory high water when coarse volume is allocated
    if (on_coarse_alloc) {
        on_coarse_alloc();
    }

    cost_c.compute_volume_packed(cfg_c, cbuf, cbuf.packed_cost);
    wta_right_from_packed(cfg_c, cbuf.packed_cost, cbuf.disparity_right);

    if (cfg_c.aggregation.enable) {
        CostAggregator agg_c;
        agg_c.aggregate_packed(cfg_c, cbuf.left_gray, cbuf.packed_cost);
    }

    const int npath = (cfg_c.sgm.paths == PathType::Path8) ? 8 : 4;
    const bool use_u16 = can_use_u16_sgm_accumulator(cfg_c, npath);

    SgmOptimizer sgm_c;
    Image32f c_best, c_second;
    if (use_u16) {
        cbuf.packed_cost32.release();
        sgm_c.optimize_packed(cfg_c, cbuf.left_gray, cbuf.packed_cost, cbuf.packed_sgm16);
        sgm_c.winner_take_all_packed(cfg_c, cbuf.packed_sgm16, cbuf.disparity, &c_best, &c_second);
    } else {
        cbuf.packed_sgm16.release();
        sgm_c.optimize_packed(cfg_c, cbuf.left_gray, cbuf.packed_cost, cbuf.packed_cost32);
        sgm_c.winner_take_all_packed(cfg_c, cbuf.packed_cost32, cbuf.disparity, &c_best, &c_second);
    }

    ConfidenceEstimator conf_c;
    conf_c.estimate(cfg_c, cbuf, c_best, c_second);

    PostProcessor post_c;
    post_c.left_right_check(cfg_c, cbuf);
    if (cfg_c.post.max_fill_gap > 0) {
        post_c.fill_holes(cfg_c, cbuf);
    }
    post_c.median(cfg_c, cbuf);

    // 4. Release coarse temporary volumes to keep retained memory minimal
    cbuf.packed_cost.release();
    cbuf.packed_sgm16.release();
    cbuf.packed_cost32.release();
    cbuf.range.release();
    cbuf.disparity_right.release();
    cbuf.census_left.clear(); cbuf.census_left.shrink_to_fit();
    cbuf.census_right.clear(); cbuf.census_right.shrink_to_fit();
    cbuf.census_left64.clear(); cbuf.census_left64.shrink_to_fit();
    cbuf.census_right64.clear(); cbuf.census_right64.shrink_to_fit();
    cbuf.left_gx.release(); cbuf.left_gy.release();
    cbuf.right_gx.release(); cbuf.right_gy.release();
    cbuf.left_gray.release(); cbuf.right_gray.release();
    cbuf.left.release(); cbuf.right.release();

    // 5. Build full-resolution guided search range
    size_t coarse_valid_count = 0;
    for (int y = 0; y < H_c; ++y) {
        for (int x = 0; x < W_c; ++x) {
            if (cbuf.disparity.at(x, y) >= 0.0f) {
                ++coarse_valid_count;
            }
        }
    }

    buf.range.allocate(W, H, cfg.min_disparity, cfg.max_disparity);
    buf.d_prior = Image32f(W, H, -1.f);
    buf.supports.clear();
    buf.support_count = 0;

    const int R = hcfg.radius;
    size_t fallback_count = 0;
    size_t full_range_count = 0;
    double sum_search_width = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 8) reduction(+:fallback_count, full_range_count, sum_search_width)
#endif
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int xc = std::max(0, std::min(x / 2, W_c - 1));
            const int yc = std::max(0, std::min(y / 2, H_c - 1));

            const int xc1 = std::min(xc + 1, W_c - 1);
            const int yc1 = std::min(yc + 1, H_c - 1);

            const float d00 = cbuf.disparity.at(xc, yc);
            const float d10 = cbuf.disparity.at(xc1, yc);
            const float d01 = cbuf.disparity.at(xc, yc1);
            const float d11 = cbuf.disparity.at(xc1, yc1);

            float min_d = 1e9f;
            float max_d = -1e9f;
            int n_valid = 0;
            for (float d : {d00, d10, d01, d11}) {
                if (d >= 0.0f) {
                    min_d = std::min(min_d, d);
                    max_d = std::max(max_d, d);
                    ++n_valid;
                }
            }

            const int glo = cfg.min_disparity;
            const int ghi = std::min(cfg.max_disparity, x + 1);

            if (n_valid > 0) {
                const float d_center = (d00 >= 0.0f) ? d00 : 0.5f * (min_d + max_d);
                const float D_center = 2.0f * d_center;
                const float D_min = 2.0f * min_d;
                const float D_max = 2.0f * max_d;

                int lo = std::min(static_cast<int>(std::floor(D_center)) - R,
                                  static_cast<int>(std::floor(D_min)) - 1);
                int hi = std::max(static_cast<int>(std::ceil(D_center)) + R + 1,
                                  static_cast<int>(std::ceil(D_max)) + 2);

                lo = std::max(lo, glo);
                hi = std::min(hi, ghi);
                if (hi < lo) hi = lo;

                buf.range.set_pixel(x, y, lo, hi);
                buf.d_prior.at(x, y) = D_center;
                sum_search_width += (hi - lo);

                if (hi - lo == ghi - glo) {
                    ++full_range_count;
                }
            } else {
                // No valid coarse neighbor: trigger fallback
                ++fallback_count;

                if (hcfg.fallback == HierarchicalFallback::NearestValid ||
                    hcfg.fallback == HierarchicalFallback::ExpandedRadius) {
                    // Search in a window of radius K=4 coarse pixels
                    constexpr int K = 4;
                    float nearest_d = -1.f;
                    int best_dist2 = 1000;

                    const int y_min = std::max(0, yc - K);
                    const int y_max = std::min(H_c - 1, yc + K);
                    const int x_min = std::max(0, xc - K);
                    const int x_max = std::min(W_c - 1, xc + K);

                    for (int sy = y_min; sy <= y_max; ++sy) {
                        for (int sx = x_min; sx <= x_max; ++sx) {
                            const float cd = cbuf.disparity.at(sx, sy);
                            if (cd >= 0.0f) {
                                const int dist2 = (sx - xc) * (sx - xc) + (sy - yc) * (sy - yc);
                                if (dist2 < best_dist2) {
                                    best_dist2 = dist2;
                                    nearest_d = cd;
                                }
                            }
                        }
                    }

                    if (nearest_d >= 0.0f) {
                        const float D_near = 2.0f * nearest_d;
                        const int R_exp = (hcfg.fallback == HierarchicalFallback::ExpandedRadius) ?
                                          (2 * R) : (R + 8);
                        int lo = std::max(glo, static_cast<int>(std::floor(D_near)) - R_exp);
                        int hi = std::min(ghi, static_cast<int>(std::ceil(D_near)) + R_exp + 1);
                        if (hi < lo) hi = lo;

                        buf.range.set_pixel(x, y, lo, hi);
                        buf.d_prior.at(x, y) = D_near;
                        sum_search_width += (hi - lo);

                        if (hi - lo == ghi - glo) {
                            ++full_range_count;
                        }
                    } else {
                        // Fallback to full range
                        buf.range.set_pixel(x, y, glo, ghi);
                        sum_search_width += (ghi - glo);
                        ++full_range_count;
                    }
                } else {
                    // HierarchicalFallback::FullRange
                    buf.range.set_pixel(x, y, glo, ghi);
                    sum_search_width += (ghi - glo);
                    ++full_range_count;
                }
            }
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (telemetry) {
        const double n_pixels = static_cast<double>(W) * H;
        const double n_coarse = static_cast<double>(W_c) * H_c;
        telemetry->avg_states_per_pixel = n_pixels > 0 ? (sum_search_width / n_pixels) : 0.0;
        telemetry->coarse_valid_ratio = n_coarse > 0 ? static_cast<float>(coarse_valid_count / n_coarse) : 0.0f;
        telemetry->fallback_pixel_ratio = n_pixels > 0 ? static_cast<float>(fallback_count / n_pixels) : 0.0f;
        telemetry->full_range_pixel_ratio = n_pixels > 0 ? static_cast<float>(full_range_count / n_pixels) : 0.0f;
        telemetry->coarse_time_ms = elapsed_ms;
    }

    return true;
}

} // namespace apg
