#include "apg_sgm/image_view.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/session.hpp"

#include <iostream>
#include <vector>
#include <cassert>

using namespace apg;

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "               TEST IMAGE VIEW & RAW MEMORY INTEROPERABILITY             " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // 1. ImageView8 basic properties & empty view
    {
        ImageView8 empty_view;
        if (!empty_view.empty()) {
            std::cerr << "Expected empty_view.empty() == true\n";
            return 1;
        }
        if (empty_view.data() != nullptr || empty_view.width() != 0) {
            std::cerr << "Expected null data and 0 width for empty view\n";
            return 1;
        }
        std::cout << "  Empty view checks: PASS\n";
    }

    // 2. Conversion from Image8
    {
        Image8 img(16, 16, 1);
        img.at(5, 7) = 142;
        ImageView8 view(img);
        if (view.empty() || view.width() != 16 || view.height() != 16 || view.stride() != 16) {
            std::cerr << "View from Image8 dimension mismatch\n";
            return 1;
        }
        if (view.at(5, 7) != 142) {
            std::cerr << "Pixel value mismatch in view from Image8\n";
            return 1;
        }
        std::cout << "  Image8 to ImageView8 conversion: PASS\n";
    }

    // 3. Raw memory with row padding (stride > width * channels)
    const int W = 32;
    const int H = 32;
    const int kPad = 8;
    const int kStride = W + kPad; // 40 bytes per row

    std::vector<uint8_t> raw_left(static_cast<size_t>(kStride) * H, 0);
    std::vector<uint8_t> raw_right(static_cast<size_t>(kStride) * H, 0);

    Image8 clean_left(W, H, 1);
    Image8 clean_right(W, H, 1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t vl = static_cast<uint8_t>((x * 13 + y * 19) % 256);
            int xr = (x >= 4) ? (x - 4) : 0;
            uint8_t vr = static_cast<uint8_t>((xr * 13 + y * 19) % 256);

            raw_left[y * kStride + x] = vl;
            raw_right[y * kStride + x] = vr;

            clean_left.at(x, y) = vl;
            clean_right.at(x, y) = vr;
        }
        // Fill padding bytes with dummy sentinel values to verify they are not read
        for (int p = 0; p < kPad; ++p) {
            raw_left[y * kStride + W + p] = 0xEE;
            raw_right[y * kStride + W + p] = 0xEE;
        }
    }

    ImageView8 view_left(raw_left.data(), W, H, kStride, 1);
    ImageView8 view_right(raw_right.data(), W, H, kStride, 1);

    // Verify to_image removes padding
    {
        Image8 extracted = view_left.to_image();
        if (extracted.width() != W || extracted.height() != H) {
            std::cerr << "Extracted image dimension mismatch\n";
            return 1;
        }
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (extracted.at(x, y) != clean_left.at(x, y)) {
                    std::cerr << "Extracted image pixel mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "  Strided ImageView8 to_image unpadding: PASS\n";
    }

    // 4. Stride validation
    {
        ImageView8 invalid_stride(raw_left.data(), W, H, W - 1, 1); // stride < width
        Status s = validate_inputs(invalid_stride, view_right);
        if (s.code != StatusCode::SizeMismatch) {
            std::cerr << "Expected SizeMismatch on invalid stride, got: " << status_code_string(s.code) << "\n";
            return 1;
        }
        std::cout << "  Invalid stride rejection: PASS\n";
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast);
    cfg.min_disparity = 0;
    cfg.max_disparity = 16;
    StereoMatcher matcher(cfg);

    // 5. Zero-copy strided matching vs clean Image8 matching (100% bit-exact parity)
    StereoResult res_clean;
    StereoResult res_strided;
    {
        Status s1 = matcher.compute_status(clean_left, clean_right, res_clean);
        if (!s1.ok()) {
            std::cerr << "compute_status with clean Image8 failed: " << s1.message << "\n";
            return 1;
        }

        Status s2 = matcher.compute_status(view_left, view_right, res_strided);
        if (!s2.ok()) {
            std::cerr << "compute_status with strided ImageView8 failed: " << s2.message << "\n";
            return 1;
        }

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (res_clean.disparity.at(x, y) != res_strided.disparity.at(x, y)) {
                    std::cerr << "Disparity mismatch between clean and strided view at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_clean.confidence.at(x, y) != res_strided.confidence.at(x, y)) {
                    std::cerr << "Confidence mismatch between clean and strided view at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_clean.invalid_reason.at(x, y) != res_strided.invalid_reason.at(x, y)) {
                    std::cerr << "Invalid reason mismatch between clean and strided view at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_clean.reliable_mask.at(x, y) != res_strided.reliable_mask.at(x, y)) {
                    std::cerr << "Reliable mask mismatch between clean and strided view at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "  Clean Image8 vs Strided ImageView8 100% bit-exact parity: PASS\n";
    }

    // 6. StereoInputView interop in StereoSession
    {
        StereoSession session(cfg);
        StereoInputView input(view_left, view_right);
        StereoResult res_session;
        Status s = session.process_status(input, res_session);
        if (!s.ok()) {
            std::cerr << "Session process_status with StereoInputView failed: " << s.message << "\n";
            return 1;
        }

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (res_session.disparity.at(x, y) != res_clean.disparity.at(x, y)) {
                    std::cerr << "Session disparity mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "  StereoSession with StereoInputView: PASS\n";
    }

    std::cout << "All ImageView & raw memory interop tests passed!\n";
    return 0;
}
