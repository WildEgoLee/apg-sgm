#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/packed_volume.hpp"

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

    // -------------------------------------------------------------
    // Test PackedCostVolume equivalence vs Dense CostComputer
    // -------------------------------------------------------------
    std::cout << "[Test PackedCostVolume Equivalence]\n";
    CostComputer cc;
    PriorEstimator pe;

    // Use mode with AD and Grad to test all cost components
    auto cfg_hq = PipelineConfig::from_mode(QualityMode::HighQuality, 32);
    PipelineBuffers buf_packed;
    cc.compute_aux(left, right, cfg_hq, buf_packed);
    pe.estimate(cfg_hq, buf_packed);

    // 1. Dense compute
    cc.compute_volume(cfg_hq, buf_packed);

    // 2. Packed compute
    PackedCostVolume16 packed_cost;
    cc.compute_volume_packed(cfg_hq, buf_packed, packed_cost);

    const int w = left.width();
    const int h = left.height();
    size_t evaluated_disparities = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = buf_packed.range.dmin.at(x, y);
            const int hi = buf_packed.range.dmax.at(x, y);
            const uint16_t* p_slice = packed_cost.slice(x, y);

            for (int d = lo; d < hi; ++d) {
                const uint16_t dense_c = buf_packed.cost.at(x, y, d);
                const uint16_t packed_c = packed_cost.at(x, y, d);
                const uint16_t slice_c = p_slice[d - lo];

                if (dense_c != packed_c) {
                    std::cerr << "Mismatch at (" << x << "," << y << ", d=" << d << "): dense="
                              << dense_c << " vs packed=" << packed_c << "\n";
                    return 1;
                }
                if (packed_c != slice_c) {
                    std::cerr << "Slice mismatch at (" << x << "," << y << ", d=" << d << "): packed="
                              << packed_c << " vs slice=" << slice_c << "\n";
                    return 1;
                }
                evaluated_disparities++;
            }
        }
    }

    const size_t dense_bytes = buf_packed.cost.bytes();
    const size_t packed_bytes = packed_cost.bytes();
    const double reduction = 100.0 * (1.0 - static_cast<double>(packed_bytes) / static_cast<double>(dense_bytes));

    std::cout << "  Evaluated " << evaluated_disparities << " packed disparity states (100% bit-exact match!)\n";
    std::cout << "  Dense bytes:  " << dense_bytes << " B\n";
    std::cout << "  Packed bytes: " << packed_bytes << " B (data=" << packed_cost.data_bytes()
              << " B, offsets=" << packed_cost.offsets_bytes() << " B)\n";
    std::cout << "  Memory reduction: " << reduction << "%\n";

    if (evaluated_disparities == 0) {
        std::cerr << "Error: 0 disparities evaluated in packed test!\n";
        return 1;
    }

    std::cout << "sanity ok\n";
    return 0;
}
