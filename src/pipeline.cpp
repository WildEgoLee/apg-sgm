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
                    if (c != kInvalidCost && c < best_cost[xr]) {
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

Status validate_config(const PipelineConfig& cfg) {
    if (cfg.max_disparity <= cfg.min_disparity) {
        return Status(StatusCode::InvalidDisparityRange, "max_disparity must be strictly greater than min_disparity");
    }
    if (cfg.cost.cost_max < 0 || cfg.cost.cost_max > 65535) {
        return Status(StatusCode::InvalidConfig, "cost_max out of valid uint16 representation range [0, 65535]");
    }
    if (cfg.sgm.P1 < 0) {
        return Status(StatusCode::InvalidConfig, "SGM P1 penalty must be non-negative");
    }
    if (cfg.sgm.P2_base <= cfg.sgm.P1) {
        return Status(StatusCode::InvalidConfig, "SGM P2_base must be strictly greater than P1");
    }
    if (cfg.sgm.grad_t1 < 0 || cfg.sgm.grad_t2 < cfg.sgm.grad_t1) {
        return Status(StatusCode::InvalidConfig, "SGM gradient thresholds must satisfy 0 <= grad_t1 <= grad_t2");
    }
    if (cfg.prior.enable && cfg.prior.search_radius < 0) {
        return Status(StatusCode::InvalidConfig, "prior search_radius must be non-negative");
    }
    if (cfg.refine.enable && cfg.refine.iterations < 0) {
        return Status(StatusCode::InvalidConfig, "refine iterations must be non-negative");
    }
    return Status::Ok();
}

Status validate_inputs(const ImageView8& left, const ImageView8& right) {
    if (left.empty() || right.empty()) {
        return Status(StatusCode::EmptyInput, "empty input image(left or right)");
    }
    if (left.width() <= 0 || left.height() <= 0) {
        return Status(StatusCode::EmptyInput, "image dimensions must be positive non-zero");
    }
    if (left.width() != right.width() || left.height() != right.height()) {
        return Status(StatusCode::SizeMismatch, "left/right size mismatch (images must be rectified and aligned)");
    }
    if (left.channels() != right.channels()) {
        return Status(StatusCode::SizeMismatch, "left/right channel count mismatch");
    }
    if (left.stride() < left.width() * left.channels() || right.stride() < right.width() * right.channels()) {
        return Status(StatusCode::SizeMismatch, "invalid row stride (stride less than row byte length)");
    }
    return Status::Ok();
}

static inline bool checked_add_u64(uint64_t& out, uint64_t a, uint64_t b) {
    if (UINT64_MAX - a < b) return false;
    out = a + b;
    return true;
}

static inline bool checked_mul_u64(uint64_t& out, uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

// Conservative workspace estimate for early budget rejection.
// Accounts for all major pipeline allocations including:
// - 6 grayscale and gradient images (left_gray, right_gray, left_gx, left_gy, right_gx, right_gy)
// - Census descriptors (left and right, 32-bit or 64-bit)
// - Pixel search range bounds (min_disp, max_disp)
// - Matching & aggregation cost volumes (packed layout + slices, or dense volumes)
// - 10 persistent float buffers (d_prior, prior_confidence, prior_spread, d_prior_min, d_prior_max,
//   disparity, d_before_refine, d_after_refine, disparity_right, confidence)
// - 2 temporary float buffers used during WTA/confidence estimation (best, second)
// - 2 8-bit post-processing masks (invalid_reason, reliable_mask)
size_t estimate_workspace_bytes(int width, int height, const PipelineConfig& cfg) {
    if (width <= 0 || height <= 0) return 0;
    const uint64_t w = static_cast<uint64_t>(width);
    const uint64_t h = static_cast<uint64_t>(height);
    uint64_t npixels = 0;
    if (!checked_mul_u64(npixels, w, h)) return SIZE_MAX;

    const int d_range = cfg.max_disparity - cfg.min_disparity;
    const uint64_t D = d_range > 0 ? static_cast<uint64_t>(d_range) : 128ULL;
    uint64_t total_voxels = 0;
    if (!checked_mul_u64(total_voxels, npixels, D)) return SIZE_MAX;

    uint64_t est = 0;
    auto add_term = [&](uint64_t term) -> bool {
        return checked_add_u64(est, est, term);
    };

    // Grayscale + gradients (6 * 1 byte per pixel: left_gray, right_gray, left_gx, left_gy, right_gx, right_gy)
    uint64_t gray_grad = 0;
    if (!checked_mul_u64(gray_grad, npixels, 6ULL) || !add_term(gray_grad)) return SIZE_MAX;

    // Census (32-bit or 64-bit for left + right)
    const uint64_t census_bytes = (cfg.cost.census == CensusType::Census9x7) ? 8ULL : 4ULL;
    uint64_t census_total = 0;
    if (!checked_mul_u64(census_total, npixels, census_bytes * 2ULL) || !add_term(census_total)) return SIZE_MAX;

    // SearchRange (min_disp + max_disp int16_t per pixel)
    uint64_t range_bytes = 0;
    if (!checked_mul_u64(range_bytes, npixels, static_cast<uint64_t>(sizeof(int16_t) * 2)) || !add_term(range_bytes)) return SIZE_MAX;

    const bool use_packed = cfg.resolves_to_packed();
    if (use_packed) {
        // Packed layout: offsets (uint32_t) + dmin (int16_t)
        uint64_t layout_bytes = 0;
        if (!checked_mul_u64(layout_bytes, npixels, static_cast<uint64_t>(sizeof(uint32_t) + sizeof(int16_t))) || !add_term(layout_bytes)) return SIZE_MAX;

        // packed_cost (uint16_t)
        uint64_t cost_bytes = 0;
        if (!checked_mul_u64(cost_bytes, total_voxels, static_cast<uint64_t>(sizeof(uint16_t))) || !add_term(cost_bytes)) return SIZE_MAX;

        // packed SGM accumulator (uint16_t or uint32_t)
        const int npath = (cfg.sgm.paths == PathType::Path8) ? 8 : 4;
        const bool use_u16 = can_use_u16_sgm_accumulator(cfg, npath);
        uint64_t sgm_bytes = 0;
        if (!checked_mul_u64(sgm_bytes, total_voxels, static_cast<uint64_t>(use_u16 ? sizeof(uint16_t) : sizeof(uint32_t))) || !add_term(sgm_bytes)) return SIZE_MAX;

        if (cfg.aggregation.enable) {
            uint64_t agg_bytes = 0;
            if (!checked_mul_u64(agg_bytes, total_voxels, static_cast<uint64_t>(sizeof(uint16_t))) || !add_term(agg_bytes)) return SIZE_MAX;
        }
    } else {
        // Dense volumes: cost (uint16_t), SGM accumulator cost32 (uint32_t)
        uint64_t cost_bytes = 0;
        if (!checked_mul_u64(cost_bytes, total_voxels, static_cast<uint64_t>(sizeof(uint16_t))) || !add_term(cost_bytes)) return SIZE_MAX;
        uint64_t cost32_bytes = 0;
        if (!checked_mul_u64(cost32_bytes, total_voxels, static_cast<uint64_t>(sizeof(uint32_t))) || !add_term(cost32_bytes)) return SIZE_MAX;
        if (cfg.aggregation.enable) {
            uint64_t agg_bytes = 0;
            if (!checked_mul_u64(agg_bytes, total_voxels, static_cast<uint64_t>(sizeof(uint16_t))) || !add_term(agg_bytes)) return SIZE_MAX;
        }
    }

    // 10 persistent Image32f buffers (d_prior, prior_confidence, prior_spread, d_prior_min, d_prior_max,
    // disparity, d_before_refine, d_after_refine, disparity_right, confidence)
    // + 2 temporary Image32f buffers during WTA/confidence (best, second)
    // Total: 12 * sizeof(float) per pixel
    uint64_t float_buffers_bytes = 0;
    if (!checked_mul_u64(float_buffers_bytes, npixels, static_cast<uint64_t>(12ULL * sizeof(float))) || !add_term(float_buffers_bytes)) return SIZE_MAX;

    // 2 8-bit masks: invalid_reason, reliable_mask
    uint64_t mask_bytes = 0;
    if (!checked_mul_u64(mask_bytes, npixels, 2ULL) || !add_term(mask_bytes)) return SIZE_MAX;

    if (est > static_cast<uint64_t>(SIZE_MAX)) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(est);
}

StereoMatcher::StereoMatcher(PipelineConfig cfg) : cfg_(std::move(cfg)) {}

Status StereoMatcher::compute_status(const ImageView8& left, const ImageView8& right, StereoResult& result,
                                    StereoWorkspace& workspace, PipelineStats* stats) const {
    const int w = left.width();
    const int h = left.height();
    const int dmax = cfg_.max_disparity;

    if (workspace.options().max_retained_bytes > 0) {
        const size_t estimated = estimate_workspace_bytes(w, h, cfg_);
        if (estimated > workspace.options().max_retained_bytes) {
            result.clear();
            return Status(StatusCode::ResourceLimitExceeded, "workspace memory budget exceeded before compute");
        }
    }

    const bool was_reuse = workspace.can_reuse(w, h, dmax);
    const int eff_threads = workspace.options().num_threads > 0 ? workspace.options().num_threads : cfg_.num_threads;

    Status s = compute_impl(left, right, workspace.buffers(), stats, eff_threads);
    if (!s.ok()) {
        result.clear();
        if (workspace.options().memory_policy == MemoryPolicy::ReleasePerFrame) {
            workspace.release();
        }
        return s;
    }

    result.disparity = workspace.buffers().disparity;
    result.confidence = workspace.buffers().confidence;
    result.invalid_reason = workspace.buffers().invalid_reason;
    result.reliable_mask = workspace.buffers().reliable_mask;

    workspace.record_frame(w, h, dmax, was_reuse);

    if (workspace.options().memory_policy == MemoryPolicy::ReleasePerFrame) {
        workspace.release();
    } else if (workspace.options().memory_policy == MemoryPolicy::BoundedReuse) {
        if (workspace.options().max_retained_bytes > 0 &&
            workspace.allocated_bytes() > workspace.options().max_retained_bytes) {
            workspace.release();
            result.clear();
            if (stats) stats->retained_workspace_bytes = 0;
            return Status(StatusCode::ResourceLimitExceeded, "workspace memory budget exceeded after compute");
        }
    }

    if (stats) {
        stats->retained_workspace_bytes = workspace.allocated_bytes();
        stats->frame_allocation_count = workspace.frame_allocation_count();
        stats->frame_reuse_count = workspace.frame_reuse_count();
        stats->allocation_count = workspace.allocation_count();
        stats->reuse_count = workspace.reuse_count();
        stats->requested_threads = workspace.options().num_threads > 0 ? workspace.options().num_threads : cfg_.num_threads;
        stats->effective_threads = eff_threads;
        stats->requested_backend = (cfg_.volume_backend == VolumeBackend::Packed) ? "Packed" :
                                   (cfg_.volume_backend == VolumeBackend::Dense) ? "Dense" : "Auto";
        stats->resolved_backend = cfg_.resolves_to_packed() ? "Packed" : "Dense";
        stats->memory_policy = (workspace.options().memory_policy == MemoryPolicy::BoundedReuse) ? "BoundedReuse" : "ReleasePerFrame";
    }

    return Status::Ok();
}

Status StereoMatcher::compute_status(const ImageView8& left, const ImageView8& right, StereoResult& result,
                                    PipelineStats* stats) const {
    StereoWorkspace default_workspace;
    return compute_status(left, right, result, default_workspace, stats);
}

bool StereoMatcher::compute(const ImageView8& left, const ImageView8& right, StereoResult& result,
                            StereoWorkspace& workspace, PipelineStats* stats) const {
    Status s = compute_status(left, right, result, workspace, stats);
    if (!s.ok()) {
        last_error_ = s.message;
        return false;
    }
    return true;
}

bool StereoMatcher::compute(const ImageView8& left, const ImageView8& right, StereoResult& result,
                            PipelineStats* stats) const {
    Status s = compute_status(left, right, result, stats);
    if (!s.ok()) {
        last_error_ = s.message;
        return false;
    }
    return true;
}

bool StereoMatcher::compute(const Image8& left, const Image8& right, PipelineBuffers& out,
                            PipelineStats* stats) const {
    Status s = compute_impl(ImageView8(left), ImageView8(right), out, stats, cfg_.num_threads);
    if (!s.ok()) {
        last_error_ = s.message;
        return false;
    }
    return true;
}

Status StereoMatcher::compute_impl(const ImageView8& left, const ImageView8& right, PipelineBuffers& out,
                                   PipelineStats* stats, int num_threads) const {
    last_error_.clear();
    Status s_in = validate_inputs(left, right);
    if (!s_in.ok()) {
        last_error_ = s_in.message;
        return s_in;
    }

    Status s_cfg = validate_config(cfg_);
    if (!s_cfg.ok()) {
        last_error_ = s_cfg.message;
        return s_cfg;
    }

    try {
#if defined(_OPENMP)
    if (num_threads > 0) omp_set_num_threads(num_threads);
    else if (cfg_.num_threads > 0) omp_set_num_threads(cfg_.num_threads);
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
        out.packed_cost.allocate_for_overwrite(layout);

        cost.compute_volume_packed(cfg_, out, out.packed_cost);
        const auto t3 = time_now();

        wta_right_from_packed_volume(cfg_, out.packed_cost, out.disparity_right);
        const auto t4 = time_now();

        agg.aggregate_packed(cfg_, out.left_gray, out.packed_cost);
        const auto t5 = time_now();

        const int npath = (cfg_.sgm.paths == PathType::Path8) ? 8 : 4;
        const bool use_u16 = can_use_u16_sgm_accumulator(cfg_, npath);

        if (use_u16) {
            out.packed_cost32.release();
            sgm.optimize_packed(cfg_, out.left_gray, out.packed_cost, out.packed_sgm16);
        } else {
            out.packed_sgm16.release();
            sgm.optimize_packed(cfg_, out.left_gray, out.packed_cost, out.packed_cost32);
        }
        const auto t6 = time_now();

        if (use_u16) {
            sgm.winner_take_all_packed(cfg_, out.packed_sgm16, out.disparity, &best, &second);
        } else {
            sgm.winner_take_all_packed(cfg_, out.packed_cost32, out.disparity, &best, &second);
        }
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
            const size_t sgm_acc_bytes = out.packed_sgm16.empty() ? out.packed_cost32.data_bytes()
                                                                  : out.packed_sgm16.data_bytes();
            const size_t lay = out.packed_cost.layout_bytes();
            const size_t cross_tmp = cfg_.aggregation.enable ? c16 : 0;
            stats->cost_bytes = c16 + lay;
            stats->aggregated_cost_bytes = sgm_acc_bytes;
            stats->estimated_peak_bytes = std::max(c16 + cross_tmp, c16 + sgm_acc_bytes) + lay;
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
        size_t valid = 0;
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

                if (!out.disparity.empty() && out.disparity.at(x, y) >= 0.0f) {
                    ++valid;
                }

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
        stats->valid_pixels = valid;
        stats->reliable_pixels = reliable;
        stats->unreliable_pixels = unreliable;
        stats->refine_changed_pixels = refine_changed;
        stats->lr_fail_pixels = lr_fail;
        stats->peak_workspace_bytes = out.allocated_bytes();

        if (n_pixels > 0) {
            stats->valid_disparity_ratio = static_cast<float>(valid) / n_pixels;
            stats->reliable_ratio = static_cast<float>(reliable) / n_pixels;
            stats->unreliable_ratio = static_cast<float>(unreliable) / n_pixels;
            stats->refine_changed_ratio = static_cast<float>(refine_changed) / n_pixels;
            stats->lr_fail_ratio = static_cast<float>(lr_fail) / n_pixels;
        }
    }

    return Status::Ok();
    } catch (const std::bad_alloc& e) {
        last_error_ = std::string("memory allocation failure: ") + e.what();
        return Status(StatusCode::AllocationFailure, last_error_);
    } catch (const std::exception& e) {
        last_error_ = std::string("internal pipeline error: ") + e.what();
        return Status(StatusCode::InternalError, last_error_);
    }
}

} // namespace apg
