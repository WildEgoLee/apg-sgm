#include "inpaint_internal.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/postprocess.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace apg;

static void test_synthetic_step_edge() {
    std::cout << "Testing synthetic step edge inpainting...\n";

    // Create a 64x32 image with a background (d=10) on left and foreground (d=40) on right
    // Step boundary is at x = 30.
    // In left image, pixels in [20, 29] are simulated as occluded (invalid = -1, reason = Occlusion).
    const int w = 64;
    const int h = 32;

    PipelineBuffers buf;
    buf.disparity = Image32f(w, h, 10.0f);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));
    buf.left_gray = Image8(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x >= 30) {
                buf.disparity.at(x, y) = 40.0f;
                buf.left_gray.at(x, y) = 200;
            } else {
                buf.disparity.at(x, y) = 10.0f;
                buf.left_gray.at(x, y) = 50;
            }
            if (x >= 20 && x < 30) {
                buf.disparity.at(x, y) = -1.0f;
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Occlusion);
            }
        }
    }

    PipelineConfig cfg;
    cfg.post.max_fill_gap = 32;
    PostProcessor post;

    // Test with ExperimentalInpaintScope
    {
        ExperimentalInpaintScope scope(InpaintMethod::Legacy, 32, true);
        PipelineBuffers test_buf = buf;
        post.fill_holes(cfg, test_buf);

        InpaintTelemetry telem = ExperimentalInpaintScope::current_telemetry();
        assert(telem.total_invalid_pixels == 10 * h);
        assert(telem.classified_occ_pixels == 10 * h);
        assert(telem.route_counts[static_cast<size_t>(FillRoute::OccBothMin)] == 10 * h);

        // All occluded pixels should receive background disparity 10.0f (since min(10, 40) = 10)
        for (int y = 0; y < h; ++y) {
            for (int x = 20; x < 30; ++x) {
                assert(std::abs(test_buf.disparity.at(x, y) - 10.0f) < 1e-4f);
            }
        }
    }

    std::cout << "[PASS] test_synthetic_step_edge\n";
}

static void test_slanted_plane_preservation() {
    std::cout << "Testing slanted plane preservation...\n";

    // Surface has continuous slant: d(x, y) = 10.0f + 0.5f * x
    const int w = 64;
    const int h = 16;

    PipelineBuffers buf;
    buf.disparity = Image32f(w, h, 0.0f);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));
    buf.left_gray = Image8(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            buf.disparity.at(x, y) = 10.0f + 0.5f * static_cast<float>(x);
            buf.left_gray.at(x, y) = 100;
            // Hole of width 4 at [20, 23] marked Mismatch
            if (x >= 20 && x <= 23) {
                buf.disparity.at(x, y) = -1.0f;
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Mismatch);
            }
        }
    }

    PipelineConfig cfg;
    cfg.post.max_fill_gap = 32;
    PostProcessor post;

    for (auto method : {InpaintMethod::Legacy, InpaintMethod::ShadowReclass}) {
        ExperimentalInpaintScope scope(method, 32, true);
        PipelineBuffers test_buf = buf;
        post.fill_holes(cfg, test_buf);

        // Linear interpolation across slanted hole should restore slope exactly:
        // dl at 19 is 10 + 0.5*19 = 19.5
        // dr at 24 is 10 + 0.5*24 = 22.0
        // for x = 20: 19.5 + 0.5 = 20.0 (matches true plane)
        for (int y = 0; y < h; ++y) {
            for (int x = 20; x <= 23; ++x) {
                float expected = 10.0f + 0.5f * static_cast<float>(x);
                assert(std::abs(test_buf.disparity.at(x, y) - expected) < 1e-3f);
            }
        }
    }

    std::cout << "[PASS] test_slanted_plane_preservation\n";
}

static void test_multithread_inpaint_consistency() {
    std::cout << "Testing multithread inpaint consistency (1 thread vs 4 threads)...\n";

    const int w = 128;
    const int h = 96;
    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t pat = static_cast<uint8_t>((x * 7 + y * 13) % 256);
            left.at(x, y) = pat;
            int xr = x - 6;
            right.at(x, y) = (xr >= 0) ? pat : 0;
        }
    }

    const VolumeBackend backends[] = {VolumeBackend::Dense, VolumeBackend::Packed};

    for (auto backend : backends) {
        PipelineConfig cfg_1 = PipelineConfig::from_mode(QualityMode::Fast, 32);
        cfg_1.volume_backend = backend;
        cfg_1.num_threads = 1;

        PipelineConfig cfg_4 = PipelineConfig::from_mode(QualityMode::Fast, 32);
        cfg_4.volume_backend = backend;
        cfg_4.num_threads = 4;

        InpaintTelemetry telem_1;
        {
            ExperimentalInpaintScope scope(InpaintMethod::Legacy, 32, true);
            StereoMatcher matcher_1(cfg_1);
            StereoResult res_1;
            Status s = matcher_1.compute_status(left, right, res_1);
            assert(s.ok());
            telem_1 = ExperimentalInpaintScope::current_telemetry();
        }

        InpaintTelemetry telem_4;
        {
            ExperimentalInpaintScope scope(InpaintMethod::Legacy, 32, true);
            StereoMatcher matcher_4(cfg_4);
            StereoResult res_4;
            Status s = matcher_4.compute_status(left, right, res_4);
            assert(s.ok());
            telem_4 = ExperimentalInpaintScope::current_telemetry();
        }

        assert(telem_1.total_invalid_pixels > 0);
        assert(telem_1.total_invalid_pixels == telem_4.total_invalid_pixels);
        assert(telem_1.classified_occ_pixels == telem_4.classified_occ_pixels);
        assert(telem_1.classified_mismatch_pixels == telem_4.classified_mismatch_pixels);
        assert(telem_1.hit_max_gap_pixels == telem_4.hit_max_gap_pixels);
        assert(telem_1.invalid_runs_count == telem_4.invalid_runs_count);
        assert(telem_1.invalid_runs_truncated == telem_4.invalid_runs_truncated);
        for (int i = 0; i < 11; ++i) {
            assert(telem_1.route_counts[i] == telem_4.route_counts[i]);
        }
        for (int i = 0; i < 10; ++i) {
            assert(telem_1.run_length_buckets[i] == telem_4.run_length_buckets[i]);
        }
    }

    std::cout << "[PASS] test_multithread_inpaint_consistency\n";
}

static void test_shadow_reclass_logic() {
    std::cout << "Testing shadow reclassification logic on depth steps...\n";

    const int w = 48;
    const int h = 16;
    PipelineBuffers buf;
    buf.disparity = Image32f(w, h, 10.0f);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));
    buf.left_gray = Image8(w, h, 1);

    // Left background d=10, right foreground d=40 (step = 30 >= 4.0)
    // Hole [15, 25] marked Mismatch
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            buf.disparity.at(x, y) = (x >= 26) ? 40.0f : 10.0f;
            buf.left_gray.at(x, y) = (x >= 26) ? 200 : 50;
            if (x >= 15 && x <= 25) {
                buf.disparity.at(x, y) = -1.0f;
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Mismatch);
            }
        }
    }

    PipelineConfig cfg;
    cfg.post.max_fill_gap = 48;
    PostProcessor post;

    // Test with ShadowReclass
    {
        ExperimentalInpaintScope scope(InpaintMethod::ShadowReclass, 48, true);
        PipelineBuffers test_buf = buf;
        post.fill_holes(cfg, test_buf);

        InpaintTelemetry telem = ExperimentalInpaintScope::current_telemetry();
        assert(telem.reclassified_shadow_pixels == 11 * h);
        assert(telem.route_counts[static_cast<size_t>(FillRoute::OccBothMin)] == 11 * h);

        // All reclassified holes should take min(10, 40) = 10.0f (background disparity)
        for (int y = 0; y < h; ++y) {
            for (int x = 15; x <= 25; ++x) {
                assert(std::abs(test_buf.disparity.at(x, y) - 10.0f) < 1e-4f);
            }
        }
    }

    // Contrast with small step (< 4.0 px): should NOT reclassify
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            buf.disparity.at(x, y) = (x >= 26) ? 12.0f : 10.0f; // step = 2 < 4.0
            buf.left_gray.at(x, y) = 100;
            if (x >= 15 && x <= 25) {
                buf.disparity.at(x, y) = -1.0f;
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Mismatch);
            }
        }
    }

    {
        ExperimentalInpaintScope scope(InpaintMethod::ShadowReclass, 48, true);
        PipelineBuffers test_buf = buf;
        post.fill_holes(cfg, test_buf);

        InpaintTelemetry telem = ExperimentalInpaintScope::current_telemetry();
        assert(telem.reclassified_shadow_pixels == 0);
        assert(telem.route_counts[static_cast<size_t>(FillRoute::MismatchLerp)] == 11 * h);
    }

    std::cout << "[PASS] test_shadow_reclass_logic\n";
}

static void test_mismatch_unfilled_telemetry_accounting() {
    std::cout << "Testing mismatch unfilled telemetry accounting...\n";

    // Setup an image with 26 mismatch holes:
    // 10 holes are in a shadow step (|dr - dl| = 30 >= 4.0), reclassified and filled.
    // 5 holes are in a smooth region and filled via lerp.
    // 11 holes exceed max_fill_gap=8 and remain unfilled!
    const int w = 64;
    const int h = 1;
    PipelineBuffers buf;
    buf.disparity = Image32f(w, h, 10.0f);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));
    buf.left_gray = Image8(w, h, 1);

    // [10, 19]: 10 holes between x=9 (d=10) and x=20 (d=40) -> shadow step -> reclassified and filled
    buf.disparity.at(9, 0) = 10.0f;
    for (int x = 10; x <= 19; ++x) {
        buf.disparity.at(x, 0) = -1.0f;
        buf.invalid_reason.at(x, 0) = static_cast<uint8_t>(InvalidReason::Mismatch);
    }
    buf.disparity.at(20, 0) = 40.0f;

    // [25, 29]: 5 holes between x=24 (d=40) and x=30 (d=41) -> smooth -> filled via lerp
    buf.disparity.at(24, 0) = 40.0f;
    for (int x = 25; x <= 29; ++x) {
        buf.disparity.at(x, 0) = -1.0f;
        buf.invalid_reason.at(x, 0) = static_cast<uint8_t>(InvalidReason::Mismatch);
    }
    buf.disparity.at(30, 0) = 41.0f;

    // [40, 50]: 11 holes between x=39 (d=41) and x=51 (d=41) with gap=8 -> gap exceeded -> remain unfilled
    buf.disparity.at(39, 0) = 41.0f;
    for (int x = 40; x <= 50; ++x) {
        buf.disparity.at(x, 0) = -1.0f;
        buf.invalid_reason.at(x, 0) = static_cast<uint8_t>(InvalidReason::Mismatch);
    }
    buf.disparity.at(51, 0) = 41.0f;

    PipelineConfig cfg;
    cfg.post.max_fill_gap = 8;
    PostProcessor post;

    {
        ExperimentalInpaintScope scope(InpaintMethod::ShadowReclass, 8, true);
        PipelineBuffers test_buf = buf;
        post.fill_holes(cfg, test_buf);

        InpaintTelemetry telem = ExperimentalInpaintScope::current_telemetry();
        assert(telem.classified_mismatch_pixels == 26);
        assert(telem.reclassified_shadow_pixels == 10);
        assert(telem.route_counts[static_cast<size_t>(FillRoute::MismatchLerp)] == 5);
        // Total filled mismatch = 10 (shadow) + 5 (lerp) = 15
        assert(telem.mismatch_filled_pixels() == 15);
        // Unfilled mismatch = 26 - 15 = 11
        assert(telem.mismatch_unfilled_pixels() == 11);
        assert(telem.route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)] == 11);
    }

    std::cout << "[PASS] test_mismatch_unfilled_telemetry_accounting\n";
}

static void test_candidate_c_multithread_parity() {
    std::cout << "Testing Candidate C multithread disparity parity (1-thread vs 4-threads)...\n";

    const int w = 160;
    const int h = 120;
    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t pat = static_cast<uint8_t>((x * 11 + y * 17) % 256);
            left.at(x, y) = pat;
            int xr = x - ((x > 80) ? 12 : 4);
            right.at(x, y) = (xr >= 0) ? pat : 0;
        }
    }

    PipelineConfig cfg_1 = PipelineConfig::from_mode(QualityMode::Fast, 32);
    cfg_1.volume_backend = VolumeBackend::Packed;
    cfg_1.num_threads = 1;
    cfg_1.post.max_fill_gap = 48;

    PipelineConfig cfg_4 = PipelineConfig::from_mode(QualityMode::Fast, 32);
    cfg_4.volume_backend = VolumeBackend::Packed;
    cfg_4.num_threads = 4;
    cfg_4.post.max_fill_gap = 48;

    StereoResult res_1, res_4;
    InpaintTelemetry telem_1, telem_4;

    {
        ExperimentalInpaintScope scope(InpaintMethod::ShadowReclass, 48, true);
        StereoMatcher matcher_1(cfg_1);
        Status s = matcher_1.compute_status(left, right, res_1);
        assert(s.ok());
        telem_1 = ExperimentalInpaintScope::current_telemetry();
    }

    {
        ExperimentalInpaintScope scope(InpaintMethod::ShadowReclass, 48, true);
        StereoMatcher matcher_4(cfg_4);
        Status s = matcher_4.compute_status(left, right, res_4);
        assert(s.ok());
        telem_4 = ExperimentalInpaintScope::current_telemetry();
    }

    assert(telem_1.total_invalid_pixels == telem_4.total_invalid_pixels);
    assert(telem_1.reclassified_shadow_pixels == telem_4.reclassified_shadow_pixels);
    assert(telem_1.reclassified_shadow_pixels > 0);

    // Verify 100% bit-exact disparity buffer parity
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float d1 = res_1.disparity.at(x, y);
            float d4 = res_4.disparity.at(x, y);
            assert(d1 == d4 || (std::isnan(d1) && std::isnan(d4)));
        }
    }

    std::cout << "[PASS] test_candidate_c_multithread_parity\n";
}

static void test_quality_v2_preset_and_native_inpaint() {
    std::cout << "Testing QualityV2 preset and native inpainting without scope...\n";

    // 1. Verify PipelineConfig::quality_v2 preset fields
    PipelineConfig cfg = PipelineConfig::quality_v2(64);
    assert(cfg.mode == QualityMode::QualityV2);
    assert(cfg.max_disparity == 64);
    assert(cfg.hierarchical.enable == true);
    assert(cfg.hierarchical.radius == 4);
    assert(cfg.hierarchical.scale == 2);
    assert(cfg.prior.enable == true);
    assert(cfg.refine.enable == true);
    assert(cfg.post.subpixel == true);
    assert(cfg.post.lr_check == true);
    assert(cfg.post.max_fill_gap == 48);
    assert(cfg.post.shadow_reclass == true);

    // 2. Verify that PostProcessor::fill_holes without ExperimentalInpaintScope
    // natively executes shadow reclassification when cfg.post.shadow_reclass is true!
    const int w = 32;
    const int h = 1;
    PipelineBuffers buf;
    buf.disparity = Image32f(w, h, 10.0f);
    buf.invalid_reason = Image8u1(w, h, static_cast<uint8_t>(InvalidReason::Valid));
    buf.left_gray = Image8(w, h, 1);

    // Create a shadow step mismatch hole: dl = 10, dr = 40 (|dr - dl| = 30 >= 4.0)
    buf.disparity.at(9, 0) = 10.0f;
    for (int x = 10; x <= 15; ++x) {
        buf.disparity.at(x, 0) = -1.0f;
        buf.invalid_reason.at(x, 0) = static_cast<uint8_t>(InvalidReason::Mismatch);
    }
    buf.disparity.at(16, 0) = 40.0f;

    PostProcessor post;
    PipelineBuffers test_buf = buf;
    post.fill_holes(cfg, test_buf);

    // Under shadow_reclass (native), holes should be filled with min(dl, dr) = 10.0f
    for (int x = 10; x <= 15; ++x) {
        assert(test_buf.disparity.at(x, 0) == 10.0f);
    }

    std::cout << "[PASS] test_quality_v2_preset_and_native_inpaint\n";
}

int main() {
    std::cout << std::unitbuf;
    test_synthetic_step_edge();
    test_slanted_plane_preservation();
    test_multithread_inpaint_consistency();
    test_shadow_reclass_logic();
    test_mismatch_unfilled_telemetry_accounting();
    test_candidate_c_multithread_parity();
    test_quality_v2_preset_and_native_inpaint();
    std::cout << "All inpainting and occlusion refinement unit tests passed successfully.\n";
    return 0;
}
