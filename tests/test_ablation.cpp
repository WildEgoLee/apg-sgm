#include "apg_sgm/ablation.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/metrics.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace apg;

static void test_single_variable_ablation() {
    auto ids = all_ablation_ids();
    assert(ids.size() == 7);

    auto cfgA = make_ablation_config(AblationId::A_Census4, 64);
    auto cfgB = make_ablation_config(AblationId::B_MultiCost4, 64);
    auto cfgC = make_ablation_config(AblationId::C_Cross4, 64);
    auto cfgD = make_ablation_config(AblationId::D_Adaptive4, 64);
    auto cfgE = make_ablation_config(AblationId::E_Prior4, 64);
    auto cfgF = make_ablation_config(AblationId::F_Refine4, 64);
    auto cfgG = make_ablation_config(AblationId::G_HQ8, 64);

    // A -> B: only Cost changes
    assert(!cfgA.cost.use_ad && !cfgA.cost.use_grad);
    assert(cfgB.cost.use_ad && cfgB.cost.use_grad);
    assert(cfgA.aggregation.enable == cfgB.aggregation.enable);
    assert(cfgA.sgm.adaptive_penalties == cfgB.sgm.adaptive_penalties);
    assert(cfgA.prior.enable == cfgB.prior.enable);
    assert(cfgA.refine.enable == cfgB.refine.enable);
    assert(cfgA.sgm.paths == cfgB.sgm.paths);

    // B -> C: only Cross Aggregation changes
    assert(!cfgB.aggregation.enable && cfgC.aggregation.enable);
    assert(cfgB.cost.use_ad == cfgC.cost.use_ad);
    assert(cfgB.sgm.adaptive_penalties == cfgC.sgm.adaptive_penalties);
    assert(cfgB.prior.enable == cfgC.prior.enable);
    assert(cfgB.refine.enable == cfgC.refine.enable);
    assert(cfgB.sgm.paths == cfgC.sgm.paths);

    // C -> D: only P2 Adaptive changes
    assert(!cfgC.sgm.adaptive_penalties && cfgD.sgm.adaptive_penalties);
    assert(cfgC.cost.use_ad == cfgD.cost.use_ad);
    assert(cfgC.aggregation.enable == cfgD.aggregation.enable);
    assert(cfgC.prior.enable == cfgD.prior.enable);
    assert(cfgC.refine.enable == cfgD.refine.enable);
    assert(cfgC.sgm.paths == cfgD.sgm.paths);

    // D -> E: only Prior changes
    assert(!cfgD.prior.enable && cfgE.prior.enable);
    assert(cfgD.cost.use_ad == cfgE.cost.use_ad);
    assert(cfgD.aggregation.enable == cfgE.aggregation.enable);
    assert(cfgD.sgm.adaptive_penalties == cfgE.sgm.adaptive_penalties);
    assert(cfgD.refine.enable == cfgE.refine.enable);
    assert(cfgD.sgm.paths == cfgE.sgm.paths);

    // E -> F: only Refine changes
    assert(!cfgE.refine.enable && cfgF.refine.enable);
    assert(cfgE.cost.use_ad == cfgF.cost.use_ad);
    assert(cfgE.aggregation.enable == cfgF.aggregation.enable);
    assert(cfgE.sgm.adaptive_penalties == cfgF.sgm.adaptive_penalties);
    assert(cfgE.prior.enable == cfgF.prior.enable);
    assert(cfgE.sgm.paths == cfgF.sgm.paths);

    // F -> G: only Paths change (4 -> 8)
    assert(cfgF.sgm.paths == PathType::Path4 && cfgG.sgm.paths == PathType::Path8);
    assert(cfgF.cost.use_ad == cfgG.cost.use_ad);
    assert(cfgF.aggregation.enable == cfgG.aggregation.enable);
    assert(cfgF.sgm.adaptive_penalties == cfgG.sgm.adaptive_penalties);
    assert(cfgF.prior.enable == cfgG.prior.enable);
    assert(cfgF.refine.enable == cfgG.refine.enable);

    std::cout << "[PASS] Single-variable ablation matrix integrity verified\n";
}

static void test_pfm_io() {
    const std::string tmp_path = "test_temp_disp.pfm";
    Image32f orig(16, 12);
    for (int y = 0; y < 12; ++y) {
        for (int x = 0; x < 16; ++x) {
            orig.at(x, y) = static_cast<float>(x * 2.5f + y * 1.25f);
        }
    }

    assert(save_pfm(tmp_path, orig));

    Image32f loaded;
    assert(load_pfm(tmp_path, loaded));
    assert(loaded.width() == 16);
    assert(loaded.height() == 12);

    for (int y = 0; y < 12; ++y) {
        for (int x = 0; x < 16; ++x) {
            assert(std::abs(loaded.at(x, y) - orig.at(x, y)) < 1e-6f);
        }
    }

    std::remove(tmp_path.c_str());
    std::cout << "[PASS] PFM I/O roundtrip verified\n";
}

static void test_metrics() {
    Image32f gt(10, 10, 5.0f);
    // Create an edge in GT: column x >= 5 has disparity 10.0
    for (int y = 0; y < 10; ++y) {
        for (int x = 5; x < 10; ++x) {
            gt.at(x, y) = 10.0f;
        }
    }

    Image32f est(10, 10, 5.0f);
    for (int y = 0; y < 10; ++y) {
        for (int x = 5; x < 10; ++x) {
            est.at(x, y) = 11.5f; // Error of 1.5 px on the right half
        }
    }

    // Test SearchRange half-open interval [dmin, dmax)
    SearchRange range;
    range.allocate(10, 10, 0, 10);
    // GT is 10.0 on right half, range is [0, 10). Should NOT recall pixels with GT == 10.0
    auto m_range = evaluate_stereo(est, gt, &range);
    assert(std::abs(m_range.range_recall_all - 0.50f) < 1e-4f);

    // Test visibility mask recall
    Image8 vis(10, 10, 255);
    for (int y = 0; y < 10; ++y) {
        for (int x = 5; x < 10; ++x) {
            vis.at(x, y) = 0; // right half is occluded
        }
    }
    auto m_vis = evaluate_stereo(est, gt, &range, nullptr, nullptr, &vis);
    assert(std::abs(m_vis.range_recall_visible - 1.0f) < 1e-4f);

    // Test decoupled Refine & Post metrics
    Image32f before(10, 10, 5.0f);
    Image32f after(10, 10, 5.0f);
    for (int y = 0; y < 10; ++y) {
        for (int x = 5; x < 10; ++x) {
            before.at(x, y) = 12.0f; // error = 2.0
            after.at(x, y) = 11.0f;  // error = 1.0 (improved by 1.0)
            est.at(x, y) = 10.5f;    // error = 0.5 (post improved by 0.5)
        }
    }
    auto m_ref = evaluate_stereo(est, gt, nullptr, &before, &after);
    assert(m_ref.has_refine_stats);
    // EPE before = 1.0, after = 0.5 => delta = -0.5
    assert(std::abs(m_ref.refine_epe_delta - (-0.5f)) < 1e-4f);
    assert(std::abs(m_ref.post_epe_delta - (-0.25f)) < 1e-4f);
    assert(std::abs(m_ref.total_epe_delta - (-0.75f)) < 1e-4f);
    assert(std::abs(m_ref.refine_improved_ratio - 0.50f) < 1e-4f);

    std::cout << "[PASS] StereoMetrics evaluation and range recall verified\n";
}

static void test_volume_backend_policy() {
    auto cfgA = make_ablation_config(AblationId::A_Census4, 64);
    auto cfgB = make_ablation_config(AblationId::B_MultiCost4, 64);
    auto cfgC = make_ablation_config(AblationId::C_Cross4, 64);
    auto cfgD = make_ablation_config(AblationId::D_Adaptive4, 64);
    auto cfgE = make_ablation_config(AblationId::E_Prior4, 64);
    auto cfgF = make_ablation_config(AblationId::F_Refine4, 64);
    auto cfgG = make_ablation_config(AblationId::G_HQ8, 64);

    // Default under Auto
    assert(cfgA.volume_backend == VolumeBackend::Auto);
    assert(!cfgA.resolves_to_packed());
    assert(!cfgB.resolves_to_packed());
    assert(!cfgC.resolves_to_packed());
    assert(!cfgD.resolves_to_packed());

    assert(cfgE.resolves_to_packed());
    assert(cfgF.resolves_to_packed());
    assert(cfgG.resolves_to_packed());

    // Explicit Dense override
    cfgE.volume_backend = VolumeBackend::Dense;
    assert(!cfgE.resolves_to_packed());
    cfgG.volume_backend = VolumeBackend::Dense;
    assert(!cfgG.resolves_to_packed());

    // Explicit Packed override
    cfgA.volume_backend = VolumeBackend::Packed;
    assert(cfgA.resolves_to_packed());
    cfgD.volume_backend = VolumeBackend::Packed;
    assert(cfgD.resolves_to_packed());

    // Legacy use_packed_volume compatibility fallback (only when Auto)
    auto cfg_legacy = make_ablation_config(AblationId::A_Census4, 64);
    assert(cfg_legacy.volume_backend == VolumeBackend::Auto);
    cfg_legacy.use_packed_volume = true;
    assert(cfg_legacy.resolves_to_packed());

    // Explicit Dense override must take precedence over legacy use_packed_volume
    cfg_legacy.volume_backend = VolumeBackend::Dense;
    assert(!cfg_legacy.resolves_to_packed());

    std::cout << "[PASS] VolumeBackend policy and resolves_to_packed verified\n";
}

int main() {
    test_single_variable_ablation();
    test_pfm_io();
    test_metrics();
    test_volume_backend_policy();
    std::cout << "All ablation/metrics tests passed!\n";
    return 0;
}
