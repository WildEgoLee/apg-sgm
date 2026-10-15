#include "subpixel_estimator.hpp"
#include "apg_sgm/pipeline.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace apg;

static void test_synthetic_cost_curves() {
    std::cout << "Testing synthetic cost curve cases...\n";

    // Case 1: perfect symmetric minimum C = [20, 10, 20]
    {
        float d_legacy = SubpixelEstimator::compute_delta(20.f, 10.f, 20.f, SubpixelMethod::Legacy);
        float d_curv   = SubpixelEstimator::compute_delta(20.f, 10.f, 20.f, SubpixelMethod::CurvGate, 4.f);
        float d_conf   = SubpixelEstimator::compute_delta(20.f, 10.f, 20.f, SubpixelMethod::ConfGate, 0.05f, 0.2f);
        float d_stab   = SubpixelEstimator::compute_delta(20.f, 10.f, 20.f, SubpixelMethod::Stabilized, 4.f);

        assert(std::abs(d_legacy) < 1e-6f);
        assert(std::abs(d_curv) < 1e-6f);
        assert(std::abs(d_conf) < 1e-6f);
        assert(std::abs(d_stab) < 1e-6f);
    }

    // Case 2: left-biased C = [12, 10, 20] -> minimum is to the left of d (delta < 0)
    {
        float d_legacy = SubpixelEstimator::compute_delta(12.f, 10.f, 20.f, SubpixelMethod::Legacy);
        float d_curv   = SubpixelEstimator::compute_delta(12.f, 10.f, 20.f, SubpixelMethod::CurvGate, 4.f);
        float d_conf   = SubpixelEstimator::compute_delta(12.f, 10.f, 20.f, SubpixelMethod::ConfGate, 0.05f, 0.2f);
        float d_stab   = SubpixelEstimator::compute_delta(12.f, 10.f, 20.f, SubpixelMethod::Stabilized, 4.f);

        assert(d_legacy < 0.0f && d_legacy >= -0.5f);
        assert(d_curv < 0.0f && d_curv >= -0.5f);
        assert(d_conf < 0.0f && d_conf >= -0.5f);
        assert(d_stab < 0.0f && d_stab >= -0.5f);
        // Stabilized should have slightly smaller magnitude due to lambda
        assert(std::abs(d_stab) < std::abs(d_legacy));
    }

    // Case 3: right-biased C = [20, 10, 12] -> minimum is to the right of d (delta > 0)
    {
        float d_legacy = SubpixelEstimator::compute_delta(20.f, 10.f, 12.f, SubpixelMethod::Legacy);
        float d_curv   = SubpixelEstimator::compute_delta(20.f, 10.f, 12.f, SubpixelMethod::CurvGate, 4.f);
        float d_conf   = SubpixelEstimator::compute_delta(20.f, 10.f, 12.f, SubpixelMethod::ConfGate, 0.05f, 0.2f);
        float d_stab   = SubpixelEstimator::compute_delta(20.f, 10.f, 12.f, SubpixelMethod::Stabilized, 4.f);

        assert(d_legacy > 0.0f && d_legacy <= 0.5f);
        assert(d_curv > 0.0f && d_curv <= 0.5f);
        assert(d_conf > 0.0f && d_conf <= 0.5f);
        assert(d_stab > 0.0f && d_stab <= 0.5f);
        assert(std::abs(d_stab) < std::abs(d_legacy));
    }

    // Case 4: flat C = [10, 10, 10] -> must not NaN/Inf, fallback to 0
    {
        float d_legacy = SubpixelEstimator::compute_delta(10.f, 10.f, 10.f, SubpixelMethod::Legacy);
        float d_curv   = SubpixelEstimator::compute_delta(10.f, 10.f, 10.f, SubpixelMethod::CurvGate);
        float d_conf   = SubpixelEstimator::compute_delta(10.f, 10.f, 10.f, SubpixelMethod::ConfGate);
        float d_stab   = SubpixelEstimator::compute_delta(10.f, 10.f, 10.f, SubpixelMethod::Stabilized);

        assert(!std::isnan(d_legacy) && !std::isinf(d_legacy) && d_legacy == 0.0f);
        assert(!std::isnan(d_curv) && !std::isinf(d_curv) && d_curv == 0.0f);
        assert(!std::isnan(d_conf) && !std::isinf(d_conf) && d_conf == 0.0f);
        assert(!std::isnan(d_stab) && !std::isinf(d_stab) && d_stab == 0.0f);
    }

    // Case 5: pathological curvature (negative curvature) C = [10, 20, 10] (local peak, not valley)
    {
        SubpixelTelemetry telem;
        float d_curv = SubpixelEstimator::compute_delta(10.f, 20.f, 10.f, SubpixelMethod::CurvGate, 4.f, 1.f, &telem);
        float d_stab = SubpixelEstimator::compute_delta(10.f, 20.f, 10.f, SubpixelMethod::Stabilized, 4.f, 1.f, &telem);

        // CurvGate and Stabilized must reject negative curvature and return 0.0
        assert(d_curv == 0.0f);
        assert(d_stab == 0.0f);
        assert(telem.negative_curvature_count > 0);
    }

    // Case 6: shallow noisy basin C = [11, 10, 11] (curvature = 2)
    {
        // Curvature is 11 - 2*10 + 11 = 2
        // If min_curvature is 4, CurvGate should reject and return 0
        float d_curv_strict = SubpixelEstimator::compute_delta(11.f, 10.f, 11.f, SubpixelMethod::CurvGate, 4.0f);
        float d_curv_loose  = SubpixelEstimator::compute_delta(11.f, 10.f, 11.f, SubpixelMethod::CurvGate, 1.0f);
        assert(d_curv_strict == 0.0f);
        assert(std::abs(d_curv_loose) < 1e-6f);
    }

    // Case 7: low uniqueness (ConfGate rejects)
    {
        // Basin is convex C = [15, 10, 20]
        float d_high_conf = SubpixelEstimator::compute_delta(15.f, 10.f, 20.f, SubpixelMethod::ConfGate, 0.10f, 0.25f);
        float d_low_conf  = SubpixelEstimator::compute_delta(15.f, 10.f, 20.f, SubpixelMethod::ConfGate, 0.10f, 0.02f);
        assert(d_high_conf < 0.0f);
        assert(d_low_conf == 0.0f); // rejected due to uniqueness < 0.10
    }

    std::cout << "[PASS] test_synthetic_cost_curves\n";
}

static void test_telemetry_accuracy() {
    std::cout << "Testing subpixel telemetry accumulation...\n";

    SubpixelTelemetry telem;
    telem.reset();

    // 1. Symmetric zero delta
    SubpixelEstimator::compute_delta(20.f, 10.f, 20.f, SubpixelMethod::Legacy, 0.f, 1.f, &telem);
    assert(telem.candidate_pixels == 1);
    assert(telem.applied_pixels == 0); // delta is 0

    // 2. Off-center delta
    SubpixelEstimator::compute_delta(14.f, 10.f, 20.f, SubpixelMethod::Legacy, 0.f, 1.f, &telem);
    assert(telem.candidate_pixels == 2);
    assert(telem.applied_pixels == 1);

    // 3. Degenerate flat
    SubpixelEstimator::compute_delta(10.f, 10.f, 10.f, SubpixelMethod::Legacy, 0.f, 1.f, &telem);
    assert(telem.candidate_pixels == 3);
    assert(telem.degenerate_count == 1);

    // Merge test
    SubpixelTelemetry telem2;
    telem2.candidate_pixels = 5;
    telem2.applied_pixels = 3;
    telem2.clamped_count = 1;
    telem2.sum_abs_delta = 0.75;

    telem.merge(telem2);
    assert(telem.candidate_pixels == 8);
    assert(telem.applied_pixels == 4);
    assert(telem.clamped_count == 1);

    std::cout << "[PASS] test_telemetry_accuracy\n";
}

static void test_scope_isolation() {
    std::cout << "Testing ExperimentalSubpixelScope isolation...\n";

    assert(ExperimentalSubpixelScope::current_method() == SubpixelMethod::Legacy);
    assert(!ExperimentalSubpixelScope::is_telemetry_enabled());

    {
        ExperimentalSubpixelScope scope1(SubpixelMethod::CurvGate, 6.0f, true);
        assert(ExperimentalSubpixelScope::current_method() == SubpixelMethod::CurvGate);
        assert(ExperimentalSubpixelScope::current_param() == 6.0f);
        assert(ExperimentalSubpixelScope::is_telemetry_enabled());

        {
            ExperimentalSubpixelScope scope2(SubpixelMethod::Stabilized, 8.0f, false);
            assert(ExperimentalSubpixelScope::current_method() == SubpixelMethod::Stabilized);
            assert(ExperimentalSubpixelScope::current_param() == 8.0f);
            assert(!ExperimentalSubpixelScope::is_telemetry_enabled());
        }

        assert(ExperimentalSubpixelScope::current_method() == SubpixelMethod::CurvGate);
        assert(ExperimentalSubpixelScope::current_param() == 6.0f);
        assert(ExperimentalSubpixelScope::is_telemetry_enabled());
    }

    assert(ExperimentalSubpixelScope::current_method() == SubpixelMethod::Legacy);
    assert(!ExperimentalSubpixelScope::is_telemetry_enabled());

    std::cout << "[PASS] test_scope_isolation\n";
}

static void test_dense_packed_subpixel_parity() {
    std::cout << "Testing Dense vs Packed dual-backend parity under subpixel methods...\n";

    const int w = 64;
    const int h = 48;
    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t v = static_cast<uint8_t>((x * 17 + y * 31) % 256);
            left.at(x, y) = v;
            int shift = (x / 16);
            int xr = std::max(0, x - shift);
            right.at(x, y) = static_cast<uint8_t>((xr * 17 + y * 31) % 256);
        }
    }

    SubpixelMethod methods[] = {
        SubpixelMethod::Legacy,
        SubpixelMethod::CurvGate,
        SubpixelMethod::ConfGate,
        SubpixelMethod::Stabilized
    };

    for (auto method : methods) {
        ExperimentalSubpixelScope scope(method, 4.0f, false);

        PipelineConfig cfg_dense = PipelineConfig::from_mode(QualityMode::Fast, 16);
        cfg_dense.volume_backend = VolumeBackend::Dense;
        cfg_dense.post.subpixel = true;

        PipelineConfig cfg_packed = PipelineConfig::from_mode(QualityMode::Fast, 16);
        cfg_packed.volume_backend = VolumeBackend::Packed;
        cfg_packed.post.subpixel = true;

        StereoMatcher matcher_dense(cfg_dense);
        StereoMatcher matcher_packed(cfg_packed);

        StereoResult res_dense, res_packed;
        Status s_d = matcher_dense.compute_status(left, right, res_dense);
        Status s_p = matcher_packed.compute_status(left, right, res_packed);

        assert(s_d.ok());
        assert(s_p.ok());

        int match_count = 0;
        int total_valid = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float dd = res_dense.disparity.at(x, y);
                float dp = res_packed.disparity.at(x, y);
                if (dd >= 0.0f && dp >= 0.0f) {
                    total_valid++;
                    if (std::abs(dd - dp) < 1e-4f) {
                        match_count++;
                    }
                }
            }
        }
        assert(total_valid > 0);
        assert(match_count == total_valid); // 100% bit-exact parity on valid pixels
    }

    std::cout << "[PASS] test_dense_packed_subpixel_parity\n";
}

static void test_multithread_telemetry_consistency() {
    std::cout << "Testing multithread telemetry consistency (1 thread vs 4 threads)...\n";

    const int w = 128;
    const int h = 96;
    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t pat = static_cast<uint8_t>((x * 7 + y * 13) % 256);
            left.at(x, y) = pat;
            int xr = x - 5;
            right.at(x, y) = (xr >= 0) ? pat : 0;
        }
    }

    const VolumeBackend backends[] = {VolumeBackend::Dense, VolumeBackend::Packed};

    for (auto backend : backends) {
        PipelineConfig cfg_1 = PipelineConfig::from_mode(QualityMode::Fast, 32);
        cfg_1.volume_backend = backend;
        cfg_1.post.subpixel = true;
        cfg_1.num_threads = 1;

        PipelineConfig cfg_4 = PipelineConfig::from_mode(QualityMode::Fast, 32);
        cfg_4.volume_backend = backend;
        cfg_4.post.subpixel = true;
        cfg_4.num_threads = 4;

        SubpixelTelemetry telem_1;
        {
            ExperimentalSubpixelScope scope(SubpixelMethod::Legacy, 0.0f, true);
            StereoMatcher matcher_1(cfg_1);
            StereoResult res_1;
            Status s = matcher_1.compute_status(left, right, res_1);
            assert(s.ok());
            telem_1 = ExperimentalSubpixelScope::current_telemetry();
        }

        SubpixelTelemetry telem_4;
        {
            ExperimentalSubpixelScope scope(SubpixelMethod::Legacy, 0.0f, true);
            StereoMatcher matcher_4(cfg_4);
            StereoResult res_4;
            Status s = matcher_4.compute_status(left, right, res_4);
            assert(s.ok());
            telem_4 = ExperimentalSubpixelScope::current_telemetry();
        }

        assert(telem_1.candidate_pixels > 0);
        assert(telem_1.candidate_pixels == telem_4.candidate_pixels);
        assert(telem_1.applied_pixels == telem_4.applied_pixels);
        assert(telem_1.degenerate_count == telem_4.degenerate_count);
        assert(telem_1.clamped_count == telem_4.clamped_count);
        assert(telem_1.negative_curvature_count == telem_4.negative_curvature_count);
        assert(std::abs(telem_1.sum_abs_delta - telem_4.sum_abs_delta) < 1e-4);
        for (int i = 0; i < 50; ++i) {
            assert(telem_1.delta_buckets[i] == telem_4.delta_buckets[i]);
        }
    }

    std::cout << "[PASS] test_multithread_telemetry_consistency\n";
}

int main() {
    std::cout << std::unitbuf;
    test_synthetic_cost_curves();
    test_telemetry_accuracy();
    test_scope_isolation();
    test_dense_packed_subpixel_parity();
    test_multithread_telemetry_consistency();
    std::cout << "All subpixel refinement tests passed successfully.\n";
    return 0;
}
