#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/cost_computer.hpp"

#include <cmath>
#include <iostream>

using namespace apg;

static uint8_t pattern(int x, int y) {
    const float s = 120.f + 70.f * std::sin(0.17f * static_cast<float>(x)) +
                    35.f * std::sin(0.23f * static_cast<float>(y)) +
                    static_cast<float>((x * 17 + y * 31) % 23);
    int v = static_cast<int>(s);
    if (x >= 28 && x < 34) v = 250;
    return static_cast<uint8_t>(clampi(v, 0, 255));
}

static Image8 make_shift_pair(int w, int h, int shift, Image8& right) {
    Image8 left(w, h, 1);
    right = Image8(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            left.at(x, y) = pattern(x, y);
            right.at(x, y) = pattern(x + shift, y);
        }
    }
    return left;
}

int main() {
    Image8 right;
    Image8 left = make_shift_pair(80, 48, 8, right);

    auto cfg = PipelineConfig::from_mode(QualityMode::Fast, 24);
    cfg.post.median_radius = 1;
    cfg.post.lr_check = true;
    StereoMatcher matcher(cfg);
    PipelineBuffers buf;
    if (!matcher.compute(left, right, buf)) {
        std::cerr << matcher.last_error() << "\n";
        return 1;
    }

    double acc = 0.0;
    int n = 0;
    int close = 0;
    for (int y = 6; y < 42; ++y) {
        for (int x = 16; x < 64; ++x) {
            const float d = buf.disparity.at(x, y);
            if (d < 0.f) continue;
            acc += d;
            ++n;
            if (std::abs(d - 8.f) < 2.f) ++close;
        }
    }
    if (n < 80) {
        std::cerr << "too few valid pixels: " << n << "\n";
        return 1;
    }
    const double mean = acc / n;
    std::cout << "mean disparity " << mean << " over " << n << " pixels, close=" << close << "\n";
    if (std::abs(mean - 8.0) > 2.5) {
        std::cerr << "mean far from expected shift 8\n";
        return 1;
    }

    const uint32_t bits = CostComputer::symmetric_census9x7(left, 20, 16);
    if (bits == 0 && CostComputer::symmetric_census9x7(left, 40, 16) == 0) {
        std::cerr << "census produced empty codes on structured pixels\n";
        return 1;
    }
    std::cout << "sanity ok\n";
    return 0;
}
