#include "apg_sgm/ablation.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/metrics.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/sgm_optimizer.hpp"

#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <string>

using namespace apg;

static uint8_t rich_texture(int x, int y) {
    const float s = 120.f + 65.f * std::sin(0.19f * static_cast<float>(x)) +
                    40.f * std::sin(0.29f * static_cast<float>(y)) +
                    static_cast<float>((x * 37 + y * 73) % 41);
    return static_cast<uint8_t>(clampi(static_cast<int>(s), 0, 255));
}

// -------------------------------------------------------------
// Unit Test: 1D SGM recurrence isolated noise suppression
// -------------------------------------------------------------
static bool test_sgm_1d_isolated_noise_suppression() {
    std::cout << "[Test 1] SGM 1D isolated perturbation suppression... ";
    const int w = 5, h = 1, D = 3;
    PipelineConfig cfg;
    cfg.sgm.P1 = 25;
    cfg.sgm.P2_base = 80;
    cfg.sgm.adaptive_penalties = false;
    cfg.sgm.paths = PathType::Path4; // we will test the horizontal path

    CostVolume base(w, h, 0, D, 0);
    // Pixel 0, 1, 3, 4: disparity 0 is clearly best (cost 10 vs 60)
    // Pixel 2: disparity 1 is locally slightly better (cost 20 vs 30)
    for (int x = 0; x < w; ++x) {
        uint16_t* s = base.slice(x, 0);
        if (x == 2) {
            s[0] = 30; // d = 0
            s[1] = 20; // d = 1 (local perturbation)
            s[2] = 70; // d = 2
        } else {
            s[0] = 10;
            s[1] = 60;
            s[2] = 70;
        }
    }

    CostVolume32 acc(w, h, 0, D, 0);
    Image8 dummy_gray(w, h, 128);
    SgmOptimizer optimizer;

    optimizer.aggregate_path(cfg, dummy_gray, base, acc, 1, 0);

    SearchRange range;
    range.allocate(w, h, 0, D);
    Image32f disp;
    optimizer.winner_take_all(cfg, acc, range, disp);

    // With large P1/P2, at pixel 2, the penalty of jumping to d=1 should outweigh the local gain of 10
    // So disp(2, 0) should remain 0.0
    const float d2 = disp.at(2, 0);
    if (std::abs(d2 - 0.0f) > 0.01f) {
        std::cerr << "FAILED: pixel 2 disparity was " << d2 << ", expected 0.0\n";
        return false;
    }
    std::cout << "PASSED (pixel 2 disp = " << d2 << ")\n";
    return true;
}

// -------------------------------------------------------------
// Synthetic Stereo Generators
// -------------------------------------------------------------
static void make_constant_disp(int w, int h, int d_val, Image8& left, Image8& right) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            left.at(x, y) = rich_texture(x, y);
            const int xr = x + d_val;
            right.at(x, y) = rich_texture(xr, y);
        }
    }
}

static void make_slanted_disp(int w, int h, Image8& left, Image8& right, Image32f& gt) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    gt = Image32f(w, h, 0.f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float d = 4.f + static_cast<float>(x) / 16.f;
            gt.at(x, y) = d;
            left.at(x, y) = rich_texture(x, y);
            const int xr = static_cast<int>(std::round(static_cast<float>(x) + d));
            right.at(x, y) = rich_texture(xr, y);
        }
    }
}

static void make_depth_step(int w, int h, int d_left, int d_right, Image8& left, Image8& right, Image32f& gt) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    gt = Image32f(w, h, 0.f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int d = (x < w / 2) ? d_left : d_right;
            gt.at(x, y) = static_cast<float>(d);
            left.at(x, y) = rich_texture(x, y);
            right.at(x, y) = rich_texture(x + d, y);
        }
    }
}

static void make_low_texture(int w, int h, int d_val, Image8& left, Image8& right) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int v = 100 + x / 3 + y / 4;
            left.at(x, y) = static_cast<uint8_t>(clampi(v, 0, 255));
            const int vr = 100 + (x + d_val) / 3 + y / 4;
            right.at(x, y) = static_cast<uint8_t>(clampi(vr, 0, 255));
        }
    }
}

static void make_brightness_shift(int w, int h, int d_val, int shift, Image8& left, Image8& right) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t val = rich_texture(x, y);
            left.at(x, y) = val;
            const uint8_t rval = rich_texture(x + d_val, y);
            right.at(x, y) = static_cast<uint8_t>(clampi(static_cast<int>(rval) + shift, 0, 255));
        }
    }
}

static uint8_t scene_repeated(int x, int y) {
    if (x < 24) return rich_texture(x, y);
    const int stripe = ((x % 8) < 4) ? 50 : 200;
    const int grad = (y * 3) % 25;
    return static_cast<uint8_t>(clampi(stripe + grad, 0, 255));
}

static void make_repeated_pattern(int w, int h, int d_val, Image8& left, Image8& right) {
    left = Image8(w, h, 1);
    right = Image8(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            left.at(x, y) = scene_repeated(x, y);
            right.at(x, y) = scene_repeated(x + d_val, y);
        }
    }
}

// -------------------------------------------------------------
// Test Runner across Fast, Balanced, HighQuality
// -------------------------------------------------------------
template <typename CheckFn>
static bool run_all_modes_test(const std::string& name,
                               const Image8& left, const Image8& right,
                               int max_disp,
                               CheckFn check_fn) {
    std::cout << "[Test " << name << "]\n";
    const QualityMode modes[] = {QualityMode::Fast, QualityMode::Balanced, QualityMode::HighQuality};
    const char* mode_names[] = {"Fast", "Balanced", "HighQuality"};

    for (int m = 0; m < 3; ++m) {
        auto cfg = PipelineConfig::from_mode(modes[m], max_disp);
        StereoMatcher matcher(cfg);
        PipelineBuffers buf;
        if (!matcher.compute(left, right, buf)) {
            std::cerr << "  FAILED in mode " << mode_names[m] << ": " << matcher.last_error() << "\n";
            return false;
        }
        if (!check_fn(mode_names[m], buf)) {
            std::cerr << "  FAILED check in mode " << mode_names[m] << "\n";
            return false;
        }
        std::cout << "  " << mode_names[m] << " mode passed.\n";
    }
    return true;
}

int main() {
    std::cout << "=== APG-SGM Comprehensive Synthetic Regression Suite ===\n";

    if (!test_sgm_1d_isolated_noise_suppression()) {
        return 1;
    }

    const int W = 100, H = 50;
    double textured_mean_conf = 0.0;

    // 1. Constant disparity d=8
    {
        Image8 left, right;
        make_constant_disp(W, H, 8, left, right);
        bool ok = run_all_modes_test("1: Constant Disparity d=8", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                int valid = 0, close = 0;
                double sum = 0.0;
                double conf_sum = 0.0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 16; x < W - 16; ++x) {
                        float d = buf.disparity.at(x, y);
                        if (d >= 0.f) {
                            valid++;
                            sum += d;
                            if (std::abs(d - 8.f) <= 1.5f) close++;
                            conf_sum += buf.confidence.at(x, y);
                        }
                    }
                }
                if (valid < 500) {
                    std::cerr << "    Too few valid pixels: " << valid << "\n";
                    return false;
                }
                double mean = sum / valid;
                double acc_rate = static_cast<double>(close) / valid;
                textured_mean_conf = conf_sum / valid;
                std::cout << "    (" << mode << ") valid=" << valid << ", mean=" << mean
                          << ", acc=" << acc_rate * 100.0 << "%, mean_conf=" << textured_mean_conf << "\n";
                return acc_rate > 0.85 && std::abs(mean - 8.0) < 1.0;
            });
        if (!ok) return 1;
    }

    // 2. Slanted plane disparity d(x) = 4 + x / 16
    {
        Image8 left, right;
        Image32f gt;
        make_slanted_disp(W, H, left, right, gt);
        bool ok = run_all_modes_test("2: Slanted Plane d(x) = 4 + x/16", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                int valid = 0, close = 0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 16; x < W - 16; ++x) {
                        float d = buf.disparity.at(x, y);
                        float g = gt.at(x, y);
                        if (d >= 0.f) {
                            valid++;
                            if (std::abs(d - g) <= 2.0f) close++;
                        }
                    }
                }
                if (valid < 500) return false;
                double acc_rate = static_cast<double>(close) / valid;
                std::cout << "    (" << mode << ") valid=" << valid << ", slanted acc=" << acc_rate * 100.0 << "%\n";
                return acc_rate > 0.80;
            });
        if (!ok) return 1;
    }

    // 3. Depth step: left d=4, right d=14
    {
        Image8 left, right;
        Image32f gt;
        make_depth_step(W, H, 4, 14, left, right, gt);
        bool ok = run_all_modes_test("3: Depth Step d=4 | d=14", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                int valid = 0, close = 0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 16; x < W - 20; ++x) {
                        if (std::abs(x - W / 2) <= 3) continue; // skip discontinuity border
                        float d = buf.disparity.at(x, y);
                        float g = gt.at(x, y);
                        if (d >= 0.f) {
                            valid++;
                            if (std::abs(d - g) <= 1.5f) close++;
                        }
                    }
                }
                if (valid < 400) return false;
                double acc_rate = static_cast<double>(close) / valid;
                std::cout << "    (" << mode << ") step acc=" << acc_rate * 100.0 << "%\n";
                return acc_rate > 0.80;
            });
        if (!ok) return 1;
    }

    // 4. Low texture ramp: verify confidence is significantly lower than textured
    {
        Image8 left, right;
        make_low_texture(W, H, 6, left, right);
        bool ok = run_all_modes_test("4: Low Texture Smooth Ramp", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                double low_conf_sum = 0.0;
                int count = 0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 10; x < W - 10; ++x) {
                        low_conf_sum += buf.confidence.at(x, y);
                        count++;
                    }
                }
                double low_mean_conf = count > 0 ? (low_conf_sum / count) : 0.0;
                std::cout << "    (" << mode << ") low texture mean conf=" << low_mean_conf
                          << " vs textured=" << textured_mean_conf << "\n";
                // Confidence on smooth ramp must be lower than textured image
                return low_mean_conf < textured_mean_conf * 0.7;
            });
        if (!ok) return 1;
    }

    // 5. Brightness shift +25
    {
        Image8 left, right;
        make_brightness_shift(W, H, 8, 25, left, right);
        bool ok = run_all_modes_test("5: Brightness Shift +25", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                int valid = 0, close = 0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 16; x < W - 16; ++x) {
                        float d = buf.disparity.at(x, y);
                        if (d >= 0.f) {
                            valid++;
                            if (std::abs(d - 8.f) <= 1.5f) close++;
                        }
                    }
                }
                double acc_rate = valid > 0 ? (static_cast<double>(close) / valid) : 0.0;
                std::cout << "    (" << mode << ") illumination shift acc=" << acc_rate * 100.0 << "%\n";
                return acc_rate > 0.80;
            });
        if (!ok) return 1;
    }

    // 6. Repeated texture: verify Bad-2(SGM) < Bad-2(raw local WTA)
    {
        Image8 left, right;
        make_repeated_pattern(W, H, 6, left, right);

        // Compute raw local WTA Bad-2 rate without SGM
        CostComputer cc;
        PipelineConfig raw_cfg = PipelineConfig::from_mode(QualityMode::Fast, 24);
        raw_cfg.aggregation.enable = false;
        PipelineBuffers raw_buf;
        cc.compute_aux(left, right, raw_cfg, raw_buf);
        PriorEstimator pe;
        pe.estimate(raw_cfg, raw_buf);
        cc.compute_volume(raw_cfg, raw_buf);
        SgmOptimizer opt;
        Image32f raw_disp;
        opt.winner_take_all(raw_cfg, raw_buf.cost, raw_buf.range, raw_disp);

        int raw_bad = 0, raw_valid = 0;
        for (int y = 6; y < H - 6; ++y) {
            for (int x = 28; x < W - 16; ++x) {
                float rd = raw_disp.at(x, y);
                if (rd >= 0.f) {
                    raw_valid++;
                    if (std::abs(rd - 6.f) > 2.0f) raw_bad++;
                }
            }
        }
        double raw_bad_rate = raw_valid > 0 ? (static_cast<double>(raw_bad) / raw_valid) : 1.0;
        std::cout << "  Raw local WTA Bad-2 in periodic region: " << raw_bad_rate * 100.0 << "%\n";

        bool ok = run_all_modes_test("6: Repeated Stripe Pattern", left, right, 24,
            [&](const char* mode, const PipelineBuffers& buf) {
                int sgm_bad = 0, sgm_valid = 0;
                for (int y = 6; y < H - 6; ++y) {
                    for (int x = 28; x < W - 16; ++x) {
                        float d = buf.disparity.at(x, y);
                        if (d >= 0.f) {
                            sgm_valid++;
                            if (std::abs(d - 6.f) > 2.0f) sgm_bad++;
                        }
                    }
                }
                double sgm_bad_rate = sgm_valid > 0 ? (static_cast<double>(sgm_bad) / sgm_valid) : 1.0;
                std::cout << "    (" << mode << ") SGM Bad-2=" << sgm_bad_rate * 100.0
                          << "% vs raw WTA=" << raw_bad_rate * 100.0 << "%\n";
                return sgm_bad_rate < raw_bad_rate;
            });
        if (!ok) return 1;
    }

    // -------------------------------------------------------------
    // Test 7: Constant Disparity Baseline-A Accuracy Gate (CI Gate)
    // -------------------------------------------------------------
    {
        std::cout << "\n[Test 7] Constant Disparity Baseline-A Accuracy Gate (CI Gate)...\n";
        const int W = 160, H = 120, D_GT = 8, DMAX = 32;
        Image8 left(W, H, 1);
        Image8 right(W, H, 1);
        Image32f gt(W, H, static_cast<float>(D_GT));

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                left.at(x, y) = rich_texture(x, y);
            }
        }
        // Physically correct left-to-right forward warp: xr = xl - d
        for (int y = 0; y < H; ++y) {
            for (int xl = 0; xl < W; ++xl) {
                const int xr = xl - D_GT;
                if (xr >= 0 && xr < W) {
                    right.at(xr, y) = left.at(xl, y);
                }
            }
        }

        // Run Baseline A (Census + 4SGM)
        auto cfgA = make_ablation_config(AblationId::A_Census4, DMAX);
        StereoMatcher matcher(cfgA);
        PipelineBuffers bufs;
        if (!matcher.compute(left, right, bufs)) {
            std::cerr << "  FAILED: matcher compute failed: " << matcher.last_error() << "\n";
            return 1;
        }

        auto m = evaluate_stereo(bufs.disparity, gt, nullptr, nullptr, nullptr, static_cast<float>(DMAX));
        std::cout << "  Baseline A on Constant Plane: EPE=" << m.epe
                  << " px, Bad-1.0=" << m.bad_1_0 << "%, Valid Ratio=" << (m.valid_ratio * 100.0f) << "%\n";

        // Assert accuracy gate
        if (m.epe > 0.5f) {
            std::cerr << "  FAILED: EPE " << m.epe << " exceeds accuracy threshold 0.5 px!\n";
            return 1;
        }
        if (m.bad_1_0 > 2.0f) {
            std::cerr << "  FAILED: Bad-1.0 " << m.bad_1_0 << "% exceeds accuracy threshold 2.0%!\n";
            return 1;
        }
        if (m.valid_ratio < 0.85f) {
            std::cerr << "  FAILED: Valid ratio " << m.valid_ratio << " is too low!\n";
            return 1;
        }
        std::cout << "  PASSED (Accuracy Gate satisfied: EPE < 0.5 px, Bad-1.0 < 2.0%)\n";
    }

    std::cout << "\n>>> ALL 8 SYNTHETIC REGRESSION TESTS PASSED! <<<\n";
    return 0;
}
