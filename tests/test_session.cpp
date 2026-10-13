#include "apg_sgm/session.hpp"
#include "apg_sgm/image.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace apg;

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "               TEST STEREO SESSION & STREAMING ARCHITECTURE               " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // 1. Session initialization
    {
        StereoSession session;
        if (session.frame_count() != 0) {
            std::cerr << "Expected initial frame_count == 0\n";
            return 1;
        }
        if (session.total_time_ms() != 0.0) {
            std::cerr << "Expected initial total_time_ms == 0\n";
            return 1;
        }
        if (session.allocated_bytes() != 0) {
            std::cerr << "Expected initial allocated_bytes == 0\n";
            return 1;
        }
        std::cout << "  Session default construction: PASS\n";
    }

    // Synthetic test images
    const int W = 32;
    const int H = 32;
    Image8 left(W, H, 1);
    Image8 right(W, H, 1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            left.at(x, y) = static_cast<uint8_t>((x * 11 + y * 17) % 256);
            int xr = (x >= 3) ? (x - 3) : 0;
            right.at(x, y) = static_cast<uint8_t>((xr * 11 + y * 17) % 256);
        }
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast);
    cfg.min_disparity = 0;
    cfg.max_disparity = 16;

    // 2. Multi-frame streaming with BoundedReuse
    ExecutionOptions opt;
    opt.memory_policy = MemoryPolicy::BoundedReuse;
    StereoSession session(cfg, opt);

    StereoResult last_result;
    size_t first_retained_bytes = 0;

    const int kStreamFrames = 5;
    for (int f = 0; f < kStreamFrames; ++f) {
        StereoResult res;
        Status s = session.process_status(left, right, res);
        if (!s.ok()) {
            std::cerr << "Stream process failed at frame " << f << ": " << s.message << "\n";
            return 1;
        }
        if (res.disparity.width() != W || res.disparity.height() != H) {
            std::cerr << "Disparity dimension mismatch at frame " << f << "\n";
            return 1;
        }

        if (f == 0) {
            first_retained_bytes = session.allocated_bytes();
            if (first_retained_bytes == 0) {
                std::cerr << "Expected non-zero retained bytes in BoundedReuse\n";
                return 1;
            }
        } else {
            // Buffer capacity should be stable in steady-state streaming of same image size
            if (session.allocated_bytes() != first_retained_bytes) {
                std::cerr << "Memory footprint changed during steady-state stream: frame "
                          << f << " bytes=" << session.allocated_bytes()
                          << " vs initial=" << first_retained_bytes << "\n";
                return 1;
            }
        }
        last_result = std::move(res);
    }

    if (session.frame_count() != kStreamFrames) {
        std::cerr << "Expected frame_count == " << kStreamFrames << ", got " << session.frame_count() << "\n";
        return 1;
    }
    if (session.total_time_ms() <= 0.0 || session.average_time_ms() <= 0.0) {
        std::cerr << "Expected positive timing stats\n";
        return 1;
    }
    std::cout << "  Multi-frame streaming (5 frames, steady-state memory): PASS\n";

    // 3. Bit-exact parity with one-shot StereoMatcher
    {
        StereoMatcher matcher(cfg);
        StereoResult one_shot_res;
        Status s = matcher.compute_status(left, right, one_shot_res);
        if (!s.ok()) {
            std::cerr << "One-shot matcher compute failed: " << s.message << "\n";
            return 1;
        }

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (last_result.disparity.at(x, y) != one_shot_res.disparity.at(x, y)) {
                    std::cerr << "Disparity mismatch between Session and Matcher at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (last_result.confidence.at(x, y) != one_shot_res.confidence.at(x, y)) {
                    std::cerr << "Confidence mismatch between Session and Matcher at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (last_result.invalid_reason.at(x, y) != one_shot_res.invalid_reason.at(x, y)) {
                    std::cerr << "Invalid reason mismatch between Session and Matcher at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (last_result.reliable_mask.at(x, y) != one_shot_res.reliable_mask.at(x, y)) {
                    std::cerr << "Reliable mask mismatch between Session and Matcher at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "  Bit-exact parity with StereoMatcher: PASS\n";
    }

    // 4. Test Session reset()
    {
        session.reset();
        if (session.frame_count() != 0) {
            std::cerr << "Expected frame_count == 0 after reset\n";
            return 1;
        }
        if (session.total_time_ms() != 0.0) {
            std::cerr << "Expected total_time_ms == 0 after reset\n";
            return 1;
        }
        if (session.allocated_bytes() != 0) {
            std::cerr << "Expected allocated_bytes == 0 after reset\n";
            return 1;
        }
        std::cout << "  Session reset & memory cleanup: PASS\n";
    }

    // 5. Error handling propagation
    {
        Image8 empty_img;
        StereoResult err_res;
        Status s = session.process_status(empty_img, right, err_res);
        if (s.code != StatusCode::EmptyInput) {
            std::cerr << "Expected EmptyInput on empty left image, got: " << status_code_string(s.code) << "\n";
            return 1;
        }
        if (session.last_error().empty()) {
            std::cerr << "Expected session.last_error() to be set on error\n";
            return 1;
        }

        bool ok = session.process(empty_img, right, err_res);
        if (ok) {
            std::cerr << "Expected bool process to return false on error\n";
            return 1;
        }
        std::cout << "  Error propagation through session: PASS\n";
    }

    std::cout << "All StereoSession tests passed!\n";
    return 0;
}
