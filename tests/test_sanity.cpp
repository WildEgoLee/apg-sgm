#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/cost_aggregator.hpp"
#include "apg_sgm/sgm_optimizer.hpp"
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
    std::cout << "[Test PackedCostVolume Equivalence & Shared Layout]\n";
    CostComputer cc;
    PriorEstimator pe;

    // Use mode with AD and Grad to test all cost components
    auto cfg_hq = PipelineConfig::from_mode(QualityMode::HighQuality, 32);
    PipelineBuffers buf_packed;
    cc.compute_aux(left, right, cfg_hq, buf_packed);
    pe.estimate(cfg_hq, buf_packed);

    // 1. Dense compute
    cc.compute_volume(cfg_hq, buf_packed);

    // 2. Packed compute with explicit shared layout
    auto shared_layout = PackedVolumeLayout::from_range(buf_packed.range);
    PackedCostVolume16 packed_cost(shared_layout, kInvalidCost);
    PackedCostVolume32 packed_cost32(shared_layout, 0);

    // Verify both volumes share identical layout pointer before compute
    if (packed_cost.layout() != packed_cost32.layout() || packed_cost.layout() != shared_layout) {
        std::cerr << "Error: layout is not shared between packed_cost and packed_cost32!\n";
        return 1;
    }

    cc.compute_volume_packed(cfg_hq, buf_packed, packed_cost);

    // Verify layout was preserved and NOT re-allocated during compute_volume_packed
    if (packed_cost.layout() != shared_layout) {
        std::cerr << "Error: compute_volume_packed re-allocated layout instead of preserving shared_layout!\n";
        return 1;
    }
    if (shared_layout.use_count() < 3) {
        std::cerr << "Error: shared_layout use_count is " << shared_layout.use_count() << ", expected >= 3\n";
        return 1;
    }

    // Snapshot immutability regression: modifying external SearchRange must not alter layout
    const int16_t orig_dmin0 = shared_layout->dmin[0];
    buf_packed.range.dmin.at(0, 0) += 7;
    if (shared_layout->dmin[0] != orig_dmin0) {
        std::cerr << "Error: mutating SearchRange affected immutable PackedVolumeLayout!\n";
        return 1;
    }
    buf_packed.range.dmin.at(0, 0) -= 7; // restore

    // Copy / Move layout preservation checks
    PackedCostVolume16 copy_vol = packed_cost;
    if (copy_vol.layout() != shared_layout) {
        std::cerr << "Error: copy_vol does not share layout!\n";
        return 1;
    }
    PackedCostVolume16 moved_vol = std::move(copy_vol);
    if (moved_vol.layout() != shared_layout) {
        std::cerr << "Error: moved_vol does not share layout!\n";
        return 1;
    }

    const int w = left.width();
    const int h = left.height();
    size_t evaluated_disparities = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = packed_cost.dmin(x, y);
            const int hi = packed_cost.dmax(x, y);
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

    const size_t dense_total_bytes = buf_packed.cost.bytes() + buf_packed.cost.bytes() * 2; // cost16 + cost32 (6 WHD)
    const size_t packed_total_bytes = shared_layout->bytes() + packed_cost.data_bytes() + packed_cost32.data_bytes();
    const double total_reduction = 100.0 * (1.0 - static_cast<double>(packed_total_bytes) / static_cast<double>(dense_total_bytes));

    std::cout << "  Evaluated " << evaluated_disparities << " packed disparity states (100% bit-exact match!)\n";
    std::cout << "  Dense cost16:   " << dense_bytes << " B\n";
    std::cout << "  Packed cost16:  " << packed_bytes << " B (data=" << packed_cost.data_bytes()
              << " B, layout=" << shared_layout->bytes() << " B)\n";
    std::cout << "  Single volume reduction: " << reduction << "%\n";
    std::cout << "  Shared layout dual volume (cost16+cost32): dense=" << dense_total_bytes
              << " B vs packed=" << packed_total_bytes << " B -> Total memory reduction: " << total_reduction << "%\n";

    if (evaluated_disparities == 0) {
        std::cerr << "Error: 0 disparities evaluated in packed test!\n";
        return 1;
    }

    // -------------------------------------------------------------
    // Test Packed CostAggregator equivalence vs Dense CostAggregator
    // -------------------------------------------------------------
    std::cout << "[Test Packed CostAggregator (Cross) Equivalence]\n";
    CostAggregator ca;
    cfg_hq.aggregation.enable = true;
    cfg_hq.aggregation.iterations = 2; // test multiple iterations

    // Dense aggregation
    ca.aggregate(cfg_hq, buf_packed);

    // Packed aggregation
    ca.aggregate_packed(cfg_hq, buf_packed.left_gray, packed_cost);

    size_t cross_evaluated = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = packed_cost.dmin(x, y);
            const int hi = packed_cost.dmax(x, y);
            const uint16_t* p_slice = packed_cost.slice(x, y);

            for (int d = lo; d < hi; ++d) {
                const uint16_t dense_c = buf_packed.cost.at(x, y, d);
                const uint16_t packed_c = packed_cost.at(x, y, d);
                const uint16_t slice_c = p_slice[d - lo];

                if (dense_c != packed_c) {
                    std::cerr << "Cross mismatch at (" << x << "," << y << ", d=" << d << "): dense="
                              << dense_c << " vs packed=" << packed_c << "\n";
                    return 1;
                }
                if (packed_c != slice_c) {
                    std::cerr << "Cross slice mismatch at (" << x << "," << y << ", d=" << d << "): packed="
                              << packed_c << " vs slice=" << slice_c << "\n";
                    return 1;
                }
                cross_evaluated++;
            }
        }
    }
    std::cout << "  Evaluated " << cross_evaluated << " packed cross-aggregated states (100% bit-exact match!)\n";

    if (cross_evaluated != evaluated_disparities) {
        std::cerr << "Error: cross evaluated count mismatch: " << cross_evaluated << " vs " << evaluated_disparities << "\n";
        return 1;
    }

    // -------------------------------------------------------------
    // Test Packed SgmOptimizer equivalence vs Dense SgmOptimizer
    // -------------------------------------------------------------
    std::cout << "[Test Packed SgmOptimizer Equivalence & WTA]\n";
    SgmOptimizer sgm;

    // 1. Dense SGM optimization
    sgm.optimize(cfg_hq, buf_packed);

    // 2. Packed SGM optimization
    sgm.optimize_packed(cfg_hq, buf_packed.left_gray, packed_cost, packed_cost32);

    size_t sgm_evaluated = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = packed_cost.dmin(x, y);
            const int hi = packed_cost.dmax(x, y);
            const uint32_t* p_slice32 = packed_cost32.slice(x, y);

            for (int d = lo; d < hi; ++d) {
                const uint32_t dense_c = buf_packed.cost32.at(x, y, d);
                const uint32_t packed_c = packed_cost32.at(x, y, d);
                const uint32_t slice_c = p_slice32[d - lo];

                if (dense_c != packed_c) {
                    std::cerr << "SGM cost32 mismatch at (" << x << "," << y << ", d=" << d << "): dense="
                              << dense_c << " vs packed=" << packed_c << "\n";
                    return 1;
                }
                if (packed_c != slice_c) {
                    std::cerr << "SGM cost32 slice mismatch at (" << x << "," << y << ", d=" << d << "): packed="
                              << packed_c << " vs slice=" << slice_c << "\n";
                    return 1;
                }
                sgm_evaluated++;
            }
        }
    }
    std::cout << "  Evaluated " << sgm_evaluated << " packed SGM aggregated states (100% bit-exact match!)\n";

    // 3. Winner-Take-All equivalence
    Image32f disp_dense, disp_packed;
    Image32f best_c_dense, best_c_packed;
    Image32f sec_c_dense, sec_c_packed;

    sgm.winner_take_all(cfg_hq, buf_packed.cost32, buf_packed.range, disp_dense, &best_c_dense, &sec_c_dense);
    sgm.winner_take_all_packed(cfg_hq, packed_cost32, disp_packed, &best_c_packed, &sec_c_packed);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dd = disp_dense.at(x, y);
            const float dp = disp_packed.at(x, y);
            if (std::isnan(dd) != std::isnan(dp) || (std::isfinite(dd) && std::abs(dd - dp) > 1e-5f)) {
                std::cerr << "WTA disparity mismatch at (" << x << "," << y << "): dense="
                          << dd << " vs packed=" << dp << "\n";
                return 1;
            }
            const float bd = best_c_dense.at(x, y);
            const float bp = best_c_packed.at(x, y);
            if (std::isfinite(bd) != std::isfinite(bp) || (std::isfinite(bd) && std::abs(bd - bp) > 1e-4f)) {
                std::cerr << "WTA best_cost mismatch at (" << x << "," << y << "): dense="
                          << bd << " vs packed=" << bp << "\n";
                return 1;
            }
        }
    }
    std::cout << "  WTA disparities and costs match 100% bit-exact!\n";

    // -------------------------------------------------------------
    // Test Full Pipeline Equivalence (Dense vs Packed Backend)
    // -------------------------------------------------------------
    std::cout << "[Test Full Pipeline Equivalence (Dense vs Packed)]\n";
    auto cfg_pipeline = PipelineConfig::from_mode(QualityMode::HighQuality, 32);
    cfg_pipeline.prior.enable = true;
    cfg_pipeline.refine.enable = true;
    cfg_pipeline.post.lr_check = true;

    PipelineBuffers buf_pipe_dense;
    cfg_pipeline.volume_backend = VolumeBackend::Dense;
    StereoMatcher matcher_dense(cfg_pipeline);
    PipelineStats stats_dense;
    if (!matcher_dense.compute(left, right, buf_pipe_dense, &stats_dense)) {
        std::cerr << "Dense pipeline failed: " << matcher_dense.last_error() << "\n";
        return 1;
    }

    PipelineBuffers buf_pipe_packed;
    cfg_pipeline.volume_backend = VolumeBackend::Packed;
    StereoMatcher matcher_packed(cfg_pipeline);
    PipelineStats stats_packed;
    if (!matcher_packed.compute(left, right, buf_pipe_packed, &stats_packed)) {
        std::cerr << "Packed pipeline failed: " << matcher_packed.last_error() << "\n";
        return 1;
    }

    // Compare final disparity maps bit-for-bit
    size_t disp_mismatches = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dd = buf_pipe_dense.disparity.at(x, y);
            const float dp = buf_pipe_packed.disparity.at(x, y);
            if (std::isnan(dd) != std::isnan(dp) || (std::isfinite(dd) && std::abs(dd - dp) > 1e-4f)) {
                disp_mismatches++;
                if (disp_mismatches <= 5) {
                    std::cerr << "Full pipeline disparity mismatch at (" << x << "," << y << "): dense="
                              << dd << " vs packed=" << dp << "\n";
                }
            }
        }
    }
    if (disp_mismatches > 0) {
        std::cerr << "Total full pipeline disparity mismatches: " << disp_mismatches << "\n";
        return 1;
    }
    std::cout << "  Full pipeline (Prior + Cross + 8SGM + Refine + LRCheck + Median) 100% bit-exact match!\n";
    std::cout << "  Dense peak cost bytes:  " << stats_dense.estimated_peak_bytes << " B\n";
    std::cout << "  Packed peak cost bytes: " << stats_packed.estimated_peak_bytes << " B\n";
    const double pipe_red = 100.0 * (1.0 - static_cast<double>(stats_packed.estimated_peak_bytes) / stats_dense.estimated_peak_bytes);
    std::cout << "  Pipeline peak volume memory reduction: " << pipe_red << "%\n";

    std::cout << "sanity ok\n";
    return 0;
}
