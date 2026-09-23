#include "apg_sgm/metrics.hpp"

#include <algorithm>
#include <cmath>

namespace apg {

SupportMetrics evaluate_supports(
    const std::vector<SupportMatch>& supports,
    const Image32f& gt,
    const Image8* vis_mask,
    float max_valid_gt,
    int cell_size) {
    SupportMetrics sm;
    sm.total = static_cast<int>(supports.size());
    if (supports.empty() || gt.empty()) return sm;

    const int w = gt.width();
    const int h = gt.height();
    const int gw = (w + cell_size - 1) / cell_size;
    const int gh = (h + cell_size - 1) / cell_size;
    const int n_cells = gw * gh;

    // Track which grid cells have visible GT
    std::vector<uint8_t> cell_has_vis_gt(n_cells, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float g = gt.at(x, y);
            if (!std::isfinite(g) || g < 0.0f || g >= max_valid_gt) continue;
            const int xr = x - static_cast<int>(std::round(g));
            const bool is_matchable = (xr >= 0 && xr < w);
            const bool is_visible = (vis_mask && !vis_mask->empty()) ? (vis_mask->at(x, y) >= 200) : is_matchable;
            if (is_visible) {
                const int gx = x / cell_size;
                const int gy = y / cell_size;
                if (gx < gw && gy < gh) {
                    cell_has_vis_gt[gy * gw + gx] = 1;
                }
            }
        }
    }
    int total_vis_cells = 0;
    for (uint8_t c : cell_has_vis_gt) total_vis_cells += c;

    std::vector<uint8_t> cell_has_correct_vis_support(n_cells, 0);
    double sum_abs_err = 0.0;
    double sum_vis_abs_err = 0.0;

    for (const auto& s : supports) {
        if (s.x < 0 || s.x >= w || s.y < 0 || s.y >= h) continue;
        const float g = gt.at(s.x, s.y);
        if (!std::isfinite(g) || g < 0.0f || g >= max_valid_gt) continue;

        sm.gt_valid++;
        const int xr = s.x - static_cast<int>(std::round(g));
        const bool is_matchable = (xr >= 0 && xr < w);
        const bool is_visible = (vis_mask && !vis_mask->empty()) ? (vis_mask->at(s.x, s.y) >= 200) : is_matchable;
        if (is_visible) sm.visible++;

        const float err = std::abs(s.disparity - g);
        sum_abs_err += err;
        if (is_visible) sum_vis_abs_err += err;

        if (err <= 0.5f) {
            sm.correct_05++;
            if (is_visible) sm.visible_correct_05++;
        }
        if (err <= 1.0f) {
            sm.correct_1++;
            if (is_visible) {
                sm.visible_correct_1++;
                const int gx = s.x / cell_size;
                const int gy = s.y / cell_size;
                if (gx < gw && gy < gh) {
                    cell_has_correct_vis_support[gy * gw + gx] = 1;
                }
            }
        }
        if (err <= 2.0f) {
            sm.correct_2++;
            if (is_visible) sm.visible_correct_2++;
        }
    }

    if (sm.gt_valid > 0) {
        sm.precision_all_05 = static_cast<float>(sm.correct_05) / static_cast<float>(sm.gt_valid);
        sm.precision_all_1  = static_cast<float>(sm.correct_1) / static_cast<float>(sm.gt_valid);
        sm.precision_all_2  = static_cast<float>(sm.correct_2) / static_cast<float>(sm.gt_valid);
        sm.precision_05 = sm.precision_all_05;
        sm.precision_1  = sm.precision_all_1;
        sm.precision_2  = sm.precision_all_2;
        sm.mae_all = static_cast<float>(sum_abs_err / sm.gt_valid);
        sm.mean_abs_error = sm.mae_all;
    }

    if (sm.visible > 0) {
        sm.precision_vis_05 = static_cast<float>(sm.visible_correct_05) / static_cast<float>(sm.visible);
        sm.precision_vis_1  = static_cast<float>(sm.visible_correct_1) / static_cast<float>(sm.visible);
        sm.precision_vis_2  = static_cast<float>(sm.visible_correct_2) / static_cast<float>(sm.visible);
        sm.mae_visible = static_cast<float>(sum_vis_abs_err / sm.visible);
    }

    int correct_cells = 0;
    for (int i = 0; i < n_cells; ++i) {
        if (cell_has_vis_gt[i] && cell_has_correct_vis_support[i]) {
            correct_cells++;
        }
    }
    if (total_vis_cells > 0) {
        sm.grid_recall_1 = static_cast<float>(correct_cells) / static_cast<float>(total_vis_cells);
        sm.grid_coverage = sm.grid_recall_1;
        sm.support_grid_recall_vis_1 = sm.grid_recall_1;
    }
    return sm;
}

StereoMetrics evaluate_stereo(
    const Image32f& est,
    const Image32f& gt,
    const SearchRange* range,
    const Image32f* d_before_refine,
    const Image32f* d_after_refine,
    const Image8* vis_mask,
    float max_valid_gt,
    const std::vector<SupportMatch>* supports) {

    StereoMetrics m;
    const int w = est.width();
    const int h = est.height();
    if (w <= 0 || h <= 0 || gt.width() != w || gt.height() != h) {
        return m;
    }

    m.total_pixels = w * h;

    double sum_epe = 0.0;
    int count_0_5 = 0;
    int count_1_0 = 0;
    int count_2_0 = 0;
    int count_3_0 = 0;
    int kitti_d1_all_count = 0;
    int kitti_d1_noc_count = 0;
    int eval_noc_count = 0;

    int lr_fail_count = 0;

    int eval_all = 0;
    int in_range_all = 0;
    int eval_matchable = 0;
    int in_range_matchable = 0;
    int eval_visible = 0;
    int in_range_visible = 0;

    int prior_miss_visible = 0;
    int prior_miss_edge = 0;
    int prior_miss_nonedge = 0;

    // Edge mask based on GT
    std::vector<uint8_t> is_edge(static_cast<size_t>(w) * h, 0);
    const int dxs[4] = {1, -1, 0, 0};
    const int dys[4] = {0, 0, 1, -1};

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float g = gt.at(x, y);
            if (!std::isfinite(g) || g < 0.0f || g >= max_valid_gt) {
                continue;
            }
            float max_diff = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dxs[k];
                const int ny = y + dys[k];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    const float ng = gt.at(nx, ny);
                    if (std::isfinite(ng) && ng >= 0.0f && ng < max_valid_gt) {
                        max_diff = std::max(max_diff, std::abs(g - ng));
                    }
                }
            }
            if (max_diff > 1.0f) {
                is_edge[static_cast<size_t>(y) * w + x] = 1;
            }
        }
    }

    double sum_edge_epe = 0.0;
    double sum_nonedge_epe = 0.0;

    // Refinement counters (strict apples-to-apples evaluation on identical pixel sets)
    double sum_before_err = 0.0;
    double sum_after_err = 0.0;
    int refine_eval_count = 0;
    int refine_improved = 0;
    int refine_worsened = 0;
    int refine_unchanged = 0;
    int refine_changed = 0;

    double sum_total_before = 0.0;
    double sum_total_after = 0.0;
    double sum_total_final = 0.0;
    int total_eval_count = 0;

    const bool has_refine_data = (d_before_refine && !d_before_refine->empty() &&
                                  d_after_refine && !d_after_refine->empty());

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float g = gt.at(x, y);
            // Strict upper bound g < max_valid_gt
            if (!std::isfinite(g) || g < 0.0f || g >= max_valid_gt) {
                continue;
            }

            m.evaluated_pixels++;

            const int xr = x - static_cast<int>(std::round(g));
            const bool is_matchable = (xr >= 0 && xr < w);
            const bool is_visible = (vis_mask && !vis_mask->empty()) ? (vis_mask->at(x, y) >= 200) : is_matchable;

            if (range) {
                m.has_range = true;
                const int dmin = range->dmin.at(x, y);
                const int dmax = range->dmax.at(x, y);
                // Strict half-open interval [dmin, dmax)
                const bool in_range = (g >= static_cast<float>(dmin) && g < static_cast<float>(dmax));

                eval_all++;
                if (in_range) in_range_all++;

                if (is_matchable) {
                    eval_matchable++;
                    if (in_range) in_range_matchable++;
                }

                if (is_visible) {
                    eval_visible++;
                    if (in_range) {
                        in_range_visible++;
                    } else {
                        prior_miss_visible++;
                        if (is_edge[static_cast<size_t>(y) * w + x]) {
                            prior_miss_edge++;
                        } else {
                            prior_miss_nonedge++;
                        }
                    }
                }
            }

            const float e = est.at(x, y);
            if (!std::isfinite(e) || e < 0.0f) {
                lr_fail_count++;
            } else {
                m.valid_pixels++;
                const float diff = std::abs(e - g);
                sum_epe += diff;

                if (diff > 0.5f) count_0_5++;
                if (diff > 1.0f) count_1_0++;
                if (diff > 2.0f) count_2_0++;
                if (diff > 3.0f) count_3_0++;

                const bool edge = (is_edge[static_cast<size_t>(y) * w + x] != 0);
                if (edge) {
                    m.edge_pixels++;
                    sum_edge_epe += diff;
                } else {
                    m.nonedge_pixels++;
                    sum_nonedge_epe += diff;
                }
            }

            const bool is_d1_outlier = (!std::isfinite(e) || e < 0.0f) ||
                (std::abs(e - g) > 3.0f && (g > 1e-3f ? (std::abs(e - g) / g > 0.05f) : true));
            if (is_d1_outlier) kitti_d1_all_count++;
            if (is_visible) {
                eval_noc_count++;
                if (is_d1_outlier) kitti_d1_noc_count++;
            }

            if (has_refine_data) {
                const float eb = d_before_refine->at(x, y);
                const float ea = d_after_refine->at(x, y);
                if (std::isfinite(eb) && eb >= 0.0f && std::isfinite(ea) && ea >= 0.0f) {
                    refine_eval_count++;
                    const float diff_b = std::abs(eb - g);
                    const float diff_a = std::abs(ea - g);
                    sum_before_err += diff_b;
                    sum_after_err += diff_a;

                    if (diff_a < diff_b - 1e-4f) {
                        refine_improved++;
                    } else if (diff_a > diff_b + 1e-4f) {
                        refine_worsened++;
                    } else {
                        refine_unchanged++;
                    }

                    if (std::abs(ea - eb) > 1e-4f) {
                        refine_changed++;
                    }

                    if (std::isfinite(e) && e >= 0.0f) {
                        total_eval_count++;
                        sum_total_before += diff_b;
                        sum_total_after += diff_a;
                        sum_total_final += std::abs(e - g);
                    }
                }
            }
        }
    }

    if (m.evaluated_pixels > 0) {
        m.valid_ratio = static_cast<float>(m.valid_pixels) / static_cast<float>(m.evaluated_pixels);
        m.lr_fail_ratio = static_cast<float>(lr_fail_count) / static_cast<float>(m.evaluated_pixels);
    }

    if (range) {
        m.has_range = true;
        m.range_recall_all = eval_all > 0 ? static_cast<float>(in_range_all) / eval_all : 0.0f;
        m.range_recall_matchable = eval_matchable > 0 ? static_cast<float>(in_range_matchable) / eval_matchable : 0.0f;
        m.range_recall_visible = eval_visible > 0 ? static_cast<float>(in_range_visible) / eval_visible : 0.0f;

        m.range_visible_eval_pixels = static_cast<size_t>(eval_visible);
        m.range_visible_in_pixels = static_cast<size_t>(in_range_visible);
        m.range_matchable_eval_pixels = static_cast<size_t>(eval_matchable);
        m.range_matchable_in_pixels = static_cast<size_t>(in_range_matchable);

        m.prior_miss_visible_count = prior_miss_visible;
        m.prior_miss_edge = prior_miss_edge;
        m.prior_miss_nonedge = prior_miss_nonedge;
        if (prior_miss_visible > 0) {
            m.prior_miss_edge_ratio = static_cast<float>(prior_miss_edge) / prior_miss_visible;
            m.prior_miss_nonedge_ratio = static_cast<float>(prior_miss_nonedge) / prior_miss_visible;
        }
    }

    if (supports && !supports->empty()) {
        m.has_supports = true;
        m.support_metrics = evaluate_supports(*supports, gt, vis_mask, max_valid_gt);
    }

    if (m.valid_pixels > 0) {
        m.epe = static_cast<float>(sum_epe / m.valid_pixels);
        m.bad_0_5 = static_cast<float>(count_0_5) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_1_0 = static_cast<float>(count_1_0) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_2_0 = static_cast<float>(count_2_0) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_3_0 = static_cast<float>(count_3_0) / static_cast<float>(m.valid_pixels) * 100.0f;
    }

    if (m.evaluated_pixels > 0) {
        m.kitti_d1_all = static_cast<float>(kitti_d1_all_count) / static_cast<float>(m.evaluated_pixels) * 100.0f;
    }
    if (eval_noc_count > 0) {
        m.kitti_d1_noc = static_cast<float>(kitti_d1_noc_count) / static_cast<float>(eval_noc_count) * 100.0f;
    }

    if (m.edge_pixels > 0) {
        m.edge_epe = static_cast<float>(sum_edge_epe / m.edge_pixels);
    }
    if (m.nonedge_pixels > 0) {
        m.nonedge_epe = static_cast<float>(sum_nonedge_epe / m.nonedge_pixels);
    }

    if (refine_eval_count > 0) {
        m.has_refine_stats = true;
        m.refine_eval_pixels = refine_eval_count;
        m.refine_epe_delta = static_cast<float>((sum_after_err - sum_before_err) / refine_eval_count);
        m.refine_improved_ratio = static_cast<float>(refine_improved) / static_cast<float>(refine_eval_count);
        m.refine_worsened_ratio = static_cast<float>(refine_worsened) / static_cast<float>(refine_eval_count);
        m.refine_unchanged_ratio = static_cast<float>(refine_unchanged) / static_cast<float>(refine_eval_count);
        m.refine_changed_ratio = static_cast<float>(refine_changed) / static_cast<float>(refine_eval_count);
    }

    if (total_eval_count > 0) {
        m.post_epe_delta = static_cast<float>((sum_total_final - sum_total_after) / total_eval_count);
        m.total_epe_delta = static_cast<float>((sum_total_final - sum_total_before) / total_eval_count);
    }

    return m;
}

} // namespace apg
