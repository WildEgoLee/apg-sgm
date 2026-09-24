#include "apg_sgm/pipeline.hpp"

#include "apg_sgm/confidence_estimator.hpp"
#include "apg_sgm/cost_aggregator.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/postprocess.hpp"
#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/refiner.hpp"
#include "apg_sgm/sgm_optimizer.hpp"

#include <chrono>
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
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
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
                    const int xl_m = xr + best_d - 1;
                    const int xl_0 = xr + best_d;
                    const int xl_p = xr + best_d + 1;
                    if (xl_m >= 0 && xl_m < w &&
                        xl_0 >= 0 && xl_0 < w &&
                        xl_p >= 0 && xl_p < w &&
                        di > 0 && di + 1 < D) {
                        const uint16_t cm = vol.slice(xl_m, y)[di - 1];
                        const uint16_t c0 = vol.slice(xl_0, y)[di];
                        const uint16_t cp = vol.slice(xl_p, y)[di + 1];
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

void wta_right_from_packed_volume(const PipelineConfig& cfg, const PackedCostVolume16& vol,
                                  Image32f& disp_right) {
    const int w = vol.width();
    const int h = vol.height();
    const int d0 = cfg.min_disparity;
    const int D = cfg.max_disparity - cfg.min_disparity;
    disp_right = Image32f(w, h, -1.f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int y = 0; y < h; ++y) {
        for (int xr = 0; xr < w; ++xr) {
            int best_d = -1;
            uint32_t best = UINT32_MAX;
            for (int d = d0; d < d0 + D; ++d) {
                const int xl = xr + d;
                if (xl < 0 || xl >= w) continue;
                if (!vol.contains(xl, y, d)) continue;
                const uint16_t c = vol.at(xl, y, d);
                if (c == kInvalidCost) continue;
                if (c < best) {
                    best = c;
                    best_d = d;
                }
            }
            if (best_d >= 0) {
                float dval = static_cast<float>(best_d);
                if (cfg.post.subpixel) {
                    const int xl_m = xr + best_d - 1;
                    const int xl_0 = xr + best_d;
                    const int xl_p = xr + best_d + 1;
                    if (xl_m >= 0 && xl_m < w &&
                        xl_0 >= 0 && xl_0 < w &&
                        xl_p >= 0 && xl_p < w &&
                        vol.contains(xl_m, y, best_d - 1) &&
                        vol.contains(xl_0, y, best_d) &&
                        vol.contains(xl_p, y, best_d + 1)) {
                        const uint16_t cm = vol.at(xl_m, y, best_d - 1);
                        const uint16_t c0 = vol.at(xl_0, y, best_d);
                        const uint16_t cp = vol.at(xl_p, y, best_d + 1);
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

} // namespace

StereoMatcher::StereoMatcher(PipelineConfig cfg) : cfg_(std::move(cfg)) {}

bool StereoMatcher::compute(const Image8& left, const Image8& right, PipelineBuffers& out,
                            PipelineStats* stats) const {
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

    auto time_now = []() { return std::chrono::steady_clock::now(); };
    auto elapsed_ms = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    const auto t_total_start = time_now();

    const auto t0 = time_now();
    cost.compute_aux(left, right, cfg_, out);
    const auto t1 = time_now();
    prior.estimate(cfg_, out);
    const auto t2 = time_now();

    Image32f best, second;

    double t_cost_ms = 0.0;
    double t_right_wta_ms = 0.0;
    double t_cross_ms = 0.0;
    double t_sgm_ms = 0.0;
    double t_wta_ms = 0.0;

    const bool use_packed = cfg_.resolves_to_packed();

    if (use_packed) {
        out.packed_cost32.release();

        auto layout = PackedVolumeLayout::from_range(out.range);
        out.packed_cost.allocate(layout, kInvalidCost);

        cost.compute_volume_packed(cfg_, out, out.packed_cost);
        const auto t3 = time_now();

        wta_right_from_packed_volume(cfg_, out.packed_cost, out.disparity_right);
        const auto t4 = time_now();

        agg.aggregate_packed(cfg_, out.left_gray, out.packed_cost);
        const auto t5 = time_now();

        out.packed_cost32.allocate(layout, 0);

        sgm.optimize_packed(cfg_, out.left_gray, out.packed_cost, out.packed_cost32);
        const auto t6 = time_now();

        sgm.winner_take_all_packed(cfg_, out.packed_cost32, out.disparity, &best, &second);
        const auto t7 = time_now();

        t_cost_ms = elapsed_ms(t2, t3);
        t_right_wta_ms = elapsed_ms(t3, t4);
        t_cross_ms = elapsed_ms(t4, t5);
        t_sgm_ms = elapsed_ms(t5, t6);
        t_wta_ms = elapsed_ms(t6, t7);
    } else {
        out.cost32.release();

        cost.compute_volume(cfg_, out);
        const auto t3 = time_now();

        // Compute right disparity from raw/local cost volume BEFORE left Cross-arm aggregation
        wta_right_from_left_volume(cfg_, out.cost, out.disparity_right);
        const auto t4 = time_now();

        agg.aggregate(cfg_, out);
        const auto t5 = time_now();

        // Optimize left volume with SGM streaming into out.cost32
        sgm.optimize(cfg_, out);
        const auto t6 = time_now();

        sgm.winner_take_all(cfg_, out.cost32, out.range, out.disparity, &best, &second);
        const auto t7 = time_now();

        t_cost_ms = elapsed_ms(t2, t3);
        t_right_wta_ms = elapsed_ms(t3, t4);
        t_cross_ms = elapsed_ms(t4, t5);
        t_sgm_ms = elapsed_ms(t5, t6);
        t_wta_ms = elapsed_ms(t6, t7);
    }

    const auto t_wta_done = time_now();
    conf.estimate(cfg_, out, best, second);
    const auto t8 = time_now();

    out.d_before_refine = out.disparity;
    refiner.refine(cfg_, out);
    out.d_after_refine = out.disparity;
    const auto t9 = time_now();

    post.left_right_check(cfg_, out);
    post.fill_holes(cfg_, out);
    post.median(cfg_, out);
    const auto t10 = time_now();

    if (stats) {
        stats->timing.aux_ms = elapsed_ms(t0, t1);
        stats->timing.prior_ms = elapsed_ms(t1, t2);
        stats->timing.cost_ms = t_cost_ms;
        stats->timing.right_wta_ms = t_right_wta_ms;
        stats->timing.cross_ms = t_cross_ms;
        stats->timing.sgm_ms = t_sgm_ms;
        stats->timing.wta_ms = t_wta_ms;
        stats->timing.confidence_ms = elapsed_ms(t_wta_done, t8);
        stats->timing.refine_ms = elapsed_ms(t8, t9);
        stats->timing.post_ms = elapsed_ms(t9, t10);
        stats->timing.total_ms = elapsed_ms(t_total_start, t10);

        if (use_packed) {
            const size_t c16 = out.packed_cost.data_bytes();
            const size_t c32 = out.packed_cost32.data_bytes();
            const size_t lay = out.packed_cost.layout_bytes();
            const size_t cross_tmp = cfg_.aggregation.enable ? c16 : 0;
            stats->cost_bytes = c16 + lay;
            stats->aggregated_cost_bytes = c32;
            stats->estimated_peak_bytes = std::max(c16 + cross_tmp, c16 + c32) + lay;
        } else {
            const size_t c16 = out.cost.bytes();
            const size_t c32 = out.cost32.bytes();
            const size_t cross_tmp = cfg_.aggregation.enable ? c16 : 0;
            stats->cost_bytes = c16;
            stats->aggregated_cost_bytes = c32;
            stats->estimated_peak_bytes = std::max(c16 + cross_tmp, c16 + c32);
        }

        const int w = out.disparity.width();
        const int h = out.disparity.height();
        const int n_pixels = w * h;
        double sum_width = 0.0;
        double sum_geom_width = 0.0;
        size_t reliable = 0;
        size_t unreliable = 0;
        size_t refine_changed = 0;
        size_t lr_fail = 0;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int lo = out.range.dmin.at(x, y);
                const int hi = out.range.dmax.at(x, y);
                sum_width += (hi > lo ? (hi - lo) : 0);

                const int glo = cfg_.min_disparity;
                const int ghi = std::min(cfg_.max_disparity, x + 1);
                sum_geom_width += (ghi > glo ? (ghi - glo) : 0);

                if (!out.reliable_mask.empty() && out.reliable_mask.at(x, y)) {
                    ++reliable;
                } else {
                    ++unreliable;
                }

                if (!out.d_before_refine.empty() && !out.d_after_refine.empty()) {
                    const float db = out.d_before_refine.at(x, y);
                    const float da = out.d_after_refine.at(x, y);
                    if (db >= 0.0f && da >= 0.0f && std::abs(da - db) > 1e-4f) {
                        ++refine_changed;
                    }
                }

                if (!out.invalid_reason.empty()) {
                    auto r = static_cast<InvalidReason>(out.invalid_reason.at(x, y));
                    if (r == InvalidReason::Mismatch || r == InvalidReason::Occlusion) {
                        ++lr_fail;
                    }
                }
            }
        }

        const int global_D = cfg_.max_disparity - cfg_.min_disparity;
        stats->prior_support_count = out.support_count;
        stats->mean_search_width = n_pixels > 0 ? (sum_width / n_pixels) : 0.0;
        stats->mean_geometry_width = n_pixels > 0 ? (sum_geom_width / n_pixels) : 0.0;
        stats->geometry_reduction_ratio = global_D > 0 ? (1.0 - stats->mean_geometry_width / global_D) : 0.0;
        stats->prior_incremental_reduction_ratio = stats->mean_geometry_width > 0.0 ?
            (1.0 - stats->mean_search_width / stats->mean_geometry_width) : 0.0;
        stats->search_reduction_ratio = global_D > 0 ? (1.0 - stats->mean_search_width / global_D) : 0.0;
        stats->reliable_pixels = reliable;
        stats->unreliable_pixels = unreliable;
        stats->refine_changed_pixels = refine_changed;
        stats->lr_fail_pixels = lr_fail;

        if (n_pixels > 0) {
            stats->reliable_ratio = static_cast<float>(reliable) / n_pixels;
            stats->unreliable_ratio = static_cast<float>(unreliable) / n_pixels;
            stats->refine_changed_ratio = static_cast<float>(refine_changed) / n_pixels;
            stats->lr_fail_ratio = static_cast<float>(lr_fail) / n_pixels;
        }
    }

    return true;
}

} // namespace apg
