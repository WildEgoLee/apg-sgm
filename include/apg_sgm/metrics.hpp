#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/types.hpp"

#include <vector>

namespace apg {

struct SupportMetrics {
    int total = 0;
    int gt_valid = 0;
    int visible = 0;
    int correct_05 = 0;
    int correct_1 = 0;
    int correct_2 = 0;
    int visible_correct_05 = 0;
    int visible_correct_1 = 0;
    int visible_correct_2 = 0;

    float precision_05 = 0.0f; // alias for precision_all_05
    float precision_1 = 0.0f;  // alias for precision_all_1
    float precision_2 = 0.0f;  // alias for precision_all_2

    float precision_all_05 = 0.0f;
    float precision_all_1 = 0.0f;
    float precision_all_2 = 0.0f;

    float precision_vis_05 = 0.0f;
    float precision_vis_1 = 0.0f;
    float precision_vis_2 = 0.0f;

    float mean_abs_error = 0.0f; // alias for mae_all
    float mae_all = 0.0f;
    float mae_visible = 0.0f;

    float grid_recall_1 = 0.0f; // fraction of visible GT cells with >=1 correct (<=1.0px) support
    float grid_coverage = 0.0f; // alias for grid_recall_1
};

struct StereoMetrics {
    int total_pixels = 0;
    int evaluated_pixels = 0;
    int valid_pixels = 0;
    float valid_ratio = 0.0f;

    float epe = 0.0f;
    float bad_0_5 = 0.0f;
    float bad_1_0 = 0.0f;
    float bad_2_0 = 0.0f;
    float bad_3_0 = 0.0f;
    float kitti_d1_all = 0.0f; // |diff| > 3.0 px AND |diff| / gt > 0.05 on all valid GT
    float kitti_d1_noc = 0.0f; // |diff| > 3.0 px AND |diff| / gt > 0.05 on non-occluded (visible) GT

    int edge_pixels = 0;
    float edge_epe = 0.0f;
    int nonedge_pixels = 0;
    float nonedge_epe = 0.0f;

    float lr_fail_ratio = 0.0f;

    // Granular SearchRange GT Recalls
    bool has_range = false;
    float range_recall_all = 0.0f;        // All evaluated GT pixels
    float range_recall_matchable = 0.0f;  // Pixels with xr = xl - round(d) in [0, W)
    float range_recall_visible = 0.0f;    // Matchable & visible in right image (z-buffer unoccluded)

    size_t range_visible_eval_pixels = 0;
    size_t range_visible_in_pixels = 0;
    size_t range_matchable_eval_pixels = 0;
    size_t range_matchable_in_pixels = 0;

    // Spatial breakdown of misses among visible GT pixels
    int prior_miss_visible_count = 0;
    int prior_miss_edge = 0;
    int prior_miss_nonedge = 0;
    float prior_miss_edge_ratio = 0.0f;
    float prior_miss_nonedge_ratio = 0.0f;

    // Support extraction metrics
    bool has_supports = false;
    SupportMetrics support_metrics;

    // Disentangled Refinement & Post-processing metrics (evaluated on identical pixel sets)
    bool has_refine_stats = false;
    int refine_eval_pixels = 0;
    float refine_epe_delta = 0.0f;       // EPE(after_refine) - EPE(before_refine)
    float post_epe_delta = 0.0f;         // EPE(final) - EPE(after_refine)
    float total_epe_delta = 0.0f;        // EPE(final) - EPE(before_refine)
    float refine_improved_ratio = 0.0f;  // fraction of pixels where error reduced
    float refine_worsened_ratio = 0.0f;  // fraction of pixels where error increased
    float refine_unchanged_ratio = 0.0f;
    float refine_changed_ratio = 0.0f;   // fraction where |d_after - d_before| > 1e-4
};

SupportMetrics evaluate_supports(
    const std::vector<SupportMatch>& supports,
    const Image32f& gt,
    const Image8* vis_mask = nullptr,
    float max_valid_gt = 1e5f,
    int cell_size = 16);

StereoMetrics evaluate_stereo(
    const Image32f& est,
    const Image32f& gt,
    const SearchRange* range = nullptr,
    const Image32f* d_before_refine = nullptr,
    const Image32f* d_after_refine = nullptr,
    const Image8* vis_mask = nullptr,
    float max_valid_gt = 1e5f,
    const std::vector<SupportMatch>* supports = nullptr);

} // namespace apg
