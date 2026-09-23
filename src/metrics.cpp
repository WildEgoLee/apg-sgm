#include "apg_sgm/metrics.hpp"

#include <cmath>
#include <algorithm>

namespace apg {

StereoMetrics evaluate_stereo(
    const Image32f& est,
    const Image32f& gt,
    const SearchRange* range,
    const Image32f* d_before_refine,
    float max_valid_gt) {

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

    int range_in_count = 0;
    int lr_fail_count = 0;

    // Edge mask based on GT
    std::vector<uint8_t> is_edge(static_cast<size_t>(w) * h, 0);
    const int dxs[4] = {1, -1, 0, 0};
    const int dys[4] = {0, 0, 1, -1};

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float g = gt.at(x, y);
            if (!std::isfinite(g) || g <= 0.0f || g > max_valid_gt) {
                continue;
            }
            float max_diff = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dxs[k];
                const int ny = y + dys[k];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    const float ng = gt.at(nx, ny);
                    if (std::isfinite(ng) && ng > 0.0f && ng <= max_valid_gt) {
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

    double sum_before_epe = 0.0;
    int refine_eval_count = 0;
    int refine_improved = 0;
    int refine_worsened = 0;
    int refine_unchanged = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float g = gt.at(x, y);
            if (!std::isfinite(g) || g <= 0.0f || g > max_valid_gt) {
                continue;
            }

            m.evaluated_pixels++;

            if (range) {
                const int dmin = range->dmin.at(x, y);
                const int dmax = range->dmax.at(x, y);
                if (g >= static_cast<float>(dmin) && g <= static_cast<float>(dmax)) {
                    range_in_count++;
                }
            }

            const float e = est.at(x, y);
            if (!std::isfinite(e) || e < 0.0f) {
                lr_fail_count++;
                continue;
            }

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

            if (d_before_refine && !d_before_refine->empty()) {
                const float eb = d_before_refine->at(x, y);
                if (std::isfinite(eb) && eb >= 0.0f) {
                    refine_eval_count++;
                    const float diff_b = std::abs(eb - g);
                    sum_before_epe += diff_b;

                    if (diff < diff_b - 1e-4f) {
                        refine_improved++;
                    } else if (diff > diff_b + 1e-4f) {
                        refine_worsened++;
                    } else {
                        refine_unchanged++;
                    }
                }
            }
        }
    }

    if (m.evaluated_pixels > 0) {
        m.valid_ratio = static_cast<float>(m.valid_pixels) / static_cast<float>(m.evaluated_pixels);
        m.lr_fail_ratio = static_cast<float>(lr_fail_count) / static_cast<float>(m.evaluated_pixels);
        if (range) {
            m.range_gt_recall = static_cast<float>(range_in_count) / static_cast<float>(m.evaluated_pixels);
        }
    }

    if (m.valid_pixels > 0) {
        m.epe = static_cast<float>(sum_epe / m.valid_pixels);
        m.bad_0_5 = static_cast<float>(count_0_5) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_1_0 = static_cast<float>(count_1_0) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_2_0 = static_cast<float>(count_2_0) / static_cast<float>(m.valid_pixels) * 100.0f;
        m.bad_3_0 = static_cast<float>(count_3_0) / static_cast<float>(m.valid_pixels) * 100.0f;
    }

    if (m.edge_pixels > 0) {
        m.edge_epe = static_cast<float>(sum_edge_epe / m.edge_pixels);
    }
    if (m.nonedge_pixels > 0) {
        m.nonedge_epe = static_cast<float>(sum_nonedge_epe / m.nonedge_pixels);
    }

    if (d_before_refine && !d_before_refine->empty() && refine_eval_count > 0) {
        m.has_refine_stats = true;
        const float before_epe = static_cast<float>(sum_before_epe / refine_eval_count);
        m.refine_epe_delta = m.epe - before_epe;
        m.refine_improved_ratio = static_cast<float>(refine_improved) / static_cast<float>(refine_eval_count);
        m.refine_worsened_ratio = static_cast<float>(refine_worsened) / static_cast<float>(refine_eval_count);
        m.refine_unchanged_ratio = static_cast<float>(refine_unchanged) / static_cast<float>(refine_eval_count);
    }

    return m;
}

} // namespace apg
