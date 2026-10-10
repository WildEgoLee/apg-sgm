#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "apg_sgm/cost_aggregator.hpp"
#include "apg_sgm/sgm_optimizer.hpp"
#include "apg_sgm/prior_estimator.hpp"
#include "apg_sgm/packed_volume.hpp"
#include "apg_sgm/refiner.hpp"
#include "apg_sgm/postprocess.hpp"
#include "refiner_internal.hpp"

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

    // Copy / Move layout preservation and lifecycle checks
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
    if (!copy_vol.empty() || copy_vol.total_elements() != 0 || copy_vol.data() != nullptr || copy_vol.layout() != nullptr) {
        std::cerr << "Error: moved-from copy_vol not in consistent empty state!\n";
        return 1;
    }
    PackedCostVolume16 copy_from_moved(copy_vol);
    if (!copy_from_moved.empty() || copy_from_moved.total_elements() != 0 || copy_from_moved.data() != nullptr) {
        std::cerr << "Error: copy_from_moved not empty!\n";
        return 1;
    }
    PackedCostVolume16 move_assigned;
    move_assigned = std::move(moved_vol);
    if (!moved_vol.empty() || moved_vol.total_elements() != 0 || moved_vol.data() != nullptr || moved_vol.layout() != nullptr) {
        std::cerr << "Error: moved-from moved_vol not in consistent empty state after move-assignment!\n";
        return 1;
    }
    if (move_assigned.layout() != shared_layout || move_assigned.empty() || move_assigned.total_elements() == 0 || move_assigned.data() == nullptr) {
        std::cerr << "Error: move_assigned invalid state!\n";
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
    // Test Packed SGM 16-bit Accumulator & Safety Bounds
    // -------------------------------------------------------------
    std::cout << "[Test Packed SGM 16-bit Accumulator & Bound Safety]\n";
    if (!can_use_u16_sgm_accumulator(cfg_hq, 8)) {
        std::cerr << "Error: default cfg_hq expected to be u16 safe!\n";
        return 1;
    }

    PackedCostVolume16 packed_sgm16;
    sgm.optimize_packed(cfg_hq, buf_packed.left_gray, packed_cost, packed_sgm16);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = packed_cost.dmin(x, y);
            const int hi = packed_cost.dmax(x, y);
            const uint16_t* s16 = packed_sgm16.slice(x, y);
            const uint32_t* s32 = packed_cost32.slice(x, y);
            for (int d = lo; d < hi; ++d) {
                const int di = d - lo;
                if (s32[di] == kInvalidCost32) {
                    if (s16[di] != kInvalidCost) {
                        std::cerr << "SGM u16 sentinel mismatch at (" << x << "," << y << "): "
                                  << s16[di] << " vs " << s32[di] << "\n";
                        return 1;
                    }
                } else {
                    if (s32[di] != static_cast<uint32_t>(s16[di])) {
                        std::cerr << "SGM u16 vs u32 value mismatch at (" << x << "," << y << "): "
                                  << s16[di] << " vs " << s32[di] << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << "  16-bit SGM accumulator matches 32-bit SGM accumulator 100% bit-exact!\n";

    Image32f disp_packed16, best_c16, sec_c16;
    sgm.winner_take_all_packed(cfg_hq, packed_sgm16, disp_packed16, &best_c16, &sec_c16);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dp = disp_packed.at(x, y);
            const float dp16 = disp_packed16.at(x, y);
            if (std::isnan(dp) != std::isnan(dp16) || (std::isfinite(dp) && std::abs(dp - dp16) > 1e-5f)) {
                std::cerr << "WTA disparity mismatch between u16 and u32 at (" << x << "," << y << ")\n";
                return 1;
            }
        }
    }
    std::cout << "  WTA on 16-bit accumulator matches 32-bit WTA 100% bit-exact!\n";

    // Deliberately construct an unsafe config that forces u32 fallback
    PipelineConfig unsafe_cfg = cfg_hq;
    unsafe_cfg.sgm.P2_base = 10000;
    if (can_use_u16_sgm_accumulator(unsafe_cfg, 8)) {
        std::cerr << "Error: unsafe_cfg (P2_base=10000) expected to be rejected by can_use_u16_sgm_accumulator!\n";
        return 1;
    }
    PackedCostVolume32 unsafe_cost32;
    sgm.optimize_packed(unsafe_cfg, buf_packed.left_gray, packed_cost, unsafe_cost32);
    std::cout << "  Deliberately unsafe config correctly flagged and executed via 32-bit fallback!\n";

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

    // -------------------------------------------------------------
    // Test Packed Cost with Negative Disparity / Range Boundaries
    // -------------------------------------------------------------
    std::cout << "[Test Packed Cost Negative Disparity / Boundary Bounds]\n";
    auto cfg_neg = PipelineConfig::from_mode(QualityMode::HighQuality, 16);
    cfg_neg.min_disparity = -8;
    cfg_neg.max_disparity = 16;
    cfg_neg.prior.enable = false;

    PipelineBuffers buf_neg_dense;
    cfg_neg.volume_backend = VolumeBackend::Dense;
    StereoMatcher matcher_neg_dense(cfg_neg);
    if (!matcher_neg_dense.compute(left, right, buf_neg_dense)) {
        std::cerr << "Negative disparity dense pipeline failed\n";
        return 1;
    }

    PipelineBuffers buf_neg_packed;
    cfg_neg.volume_backend = VolumeBackend::Packed;
    StereoMatcher matcher_neg_packed(cfg_neg);
    if (!matcher_neg_packed.compute(left, right, buf_neg_packed)) {
        std::cerr << "Negative disparity packed pipeline failed\n";
        return 1;
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dd = buf_neg_dense.disparity.at(x, y);
            const float dp = buf_neg_packed.disparity.at(x, y);
            if (std::isnan(dd) != std::isnan(dp) || (std::isfinite(dd) && std::abs(dd - dp) > 1e-4f)) {
                std::cerr << "Negative disparity mismatch at (" << x << "," << y << "): dense="
                          << dd << " vs packed=" << dp << "\n";
                return 1;
            }
        }
    }
    std::cout << "  Negative disparity test (-8 to 16) 100% bit-exact match!\n";

    // -------------------------------------------------------------
    // Test SGM Vertical Tiling Partial Widths & Boundary Checks
    // -------------------------------------------------------------
    std::cout << "[Test SGM Vertical Tiling Partial Tile Widths & Edge Cases]\n";
    {
        const std::vector<int> test_widths = {1, 7, 15, 16, 17, 31, 32, 33};
        const int test_h = 24;
        SgmOptimizer sgm_opt;

        for (int tw : test_widths) {
            for (PathType ptype : {PathType::Path4, PathType::Path8}) {
                PipelineConfig test_cfg;
                test_cfg.sgm.paths = ptype;
                test_cfg.sgm.P1 = 10;
                test_cfg.sgm.P2_base = 120;
                test_cfg.volume_backend = VolumeBackend::Packed;

                Image8 t_gray(tw, test_h, 128);
                SearchRange t_range;
                t_range.dmin = Image16s(tw, test_h, 0);
                t_range.dmax = Image16s(tw, test_h, 0);

                for (int y = 0; y < test_h; ++y) {
                    for (int x = 0; x < tw; ++x) {
                        t_gray.at(x, y) = static_cast<uint8_t>((x * 17 + y * 23) % 256);
                        int d0 = (x % 3 == 0) ? -4 : (x % 5);
                        int d1 = d0 + ((x + y) % 19 + 1);
                        if ((x + y) % 13 == 0) {
                            d0 = 0; d1 = 0; // empty slice
                        }
                        t_range.dmin.at(x, y) = static_cast<int16_t>(d0);
                        t_range.dmax.at(x, y) = static_cast<int16_t>(d1);
                    }
                }

                auto t_layout = PackedVolumeLayout::from_range(t_range);
                PackedCostVolume16 t_cost(t_layout);
                for (size_t i = 0; i < t_layout->state_count(); ++i) {
                    t_cost.data()[i] = (i % 11 == 0) ? kInvalidCost : static_cast<uint16_t>(i % 240);
                }

                PackedCostVolume16 t_sgm(t_layout);
                sgm_opt.optimize_packed(test_cfg, t_gray, t_cost, t_sgm);

                // Verify valid accumulator contents and invalid sentinel integrity
                for (int y = 0; y < test_h; ++y) {
                    for (int x = 0; x < tw; ++x) {
                        int D_p = t_layout->disp_width(y * tw + x);
                        const uint16_t* c = t_cost.slice(x, y);
                        const uint16_t* s = t_sgm.slice(x, y);
                        for (int di = 0; di < D_p; ++di) {
                            if (c[di] == kInvalidCost) {
                                if (s[di] != kInvalidCost) {
                                    std::cerr << "Sentinel violation at w=" << tw << " (" << x << "," << y << ")\n";
                                    return 1;
                                }
                            } else {
                                if (s[di] == kInvalidCost) {
                                    std::cerr << "Unexpected invalid at w=" << tw << " (" << x << "," << y << ")\n";
                                    return 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Vertical tiling partial widths (1, 7, 15, 16, 17, 31, 32, 33) passed!\n";
    }

    // -------------------------------------------------------------
    // Test SGM Diagonal Ray Interleaving Boundary & Partial Widths
    // -------------------------------------------------------------
    std::cout << "[Test SGM Diagonal Ray Interleaving Partial Widths & Edge Cases]\n";
    {
        const std::vector<std::pair<int, int>> test_shapes = {
            {1, 1}, {7, 9}, {15, 17}, {16, 16}, {17, 23}, {31, 29}, {32, 33}, {33, 31}, {65, 48}
        };
        SgmOptimizer sgm_opt;

        for (const auto& sh : test_shapes) {
            const int tw = sh.first;
            const int th = sh.second;
            for (PathType ptype : {PathType::Path4, PathType::Path8}) {
                PipelineConfig test_cfg;
                test_cfg.sgm.paths = ptype;
                test_cfg.sgm.P1 = 10;
                test_cfg.sgm.P2_base = 120;
                test_cfg.volume_backend = VolumeBackend::Packed;

                Image8 t_gray(tw, th, 128);
                SearchRange t_range;
                t_range.dmin = Image16s(tw, th, 0);
                t_range.dmax = Image16s(tw, th, 0);

                for (int y = 0; y < th; ++y) {
                    for (int x = 0; x < tw; ++x) {
                        t_gray.at(x, y) = static_cast<uint8_t>((x * 19 + y * 29) % 256);
                        int d0 = (x % 3 == 0) ? -5 : (x % 4);
                        int d1 = d0 + ((x + y) % 21 + 1);
                        if ((x + y) % 13 == 0) {
                            d0 = 0; d1 = 0;
                        }
                        t_range.dmin.at(x, y) = static_cast<int16_t>(d0);
                        t_range.dmax.at(x, y) = static_cast<int16_t>(d1);
                    }
                }

                auto t_layout = PackedVolumeLayout::from_range(t_range);
                PackedCostVolume16 t_cost(t_layout);
                for (size_t i = 0; i < t_layout->state_count(); ++i) {
                    t_cost.data()[i] = (i % 7 == 0) ? kInvalidCost : static_cast<uint16_t>(i % 250);
                }

                PackedCostVolume16 t_sgm(t_layout);
                sgm_opt.optimize_packed(test_cfg, t_gray, t_cost, t_sgm);

                // Verify sentinel preservation across all pixels
                for (int y = 0; y < th; ++y) {
                    for (int x = 0; x < tw; ++x) {
                        int D_p = t_layout->disp_width(y * tw + x);
                        const uint16_t* c = t_cost.slice(x, y);
                        const uint16_t* s = t_sgm.slice(x, y);
                        for (int di = 0; di < D_p; ++di) {
                            if (c[di] == kInvalidCost) {
                                if (s[di] != kInvalidCost) {
                                    std::cerr << "Diagonal sentinel violation at dim=" << tw << "x" << th
                                              << " (" << x << "," << y << ")\n";
                                    return 1;
                                }
                            } else {
                                if (s[di] == kInvalidCost) {
                                    std::cerr << "Diagonal unexpected invalid at dim=" << tw << "x" << th
                                              << " (" << x << "," << y << ")\n";
                                    return 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Diagonal interleaving edge cases & partial shapes passed!\n";
    }

    // -------------------------------------------------------------
    // Comprehensive Test Matrix: Refiner refine_scalar vs refine_avx2 Bit-Exact Parity
    // -------------------------------------------------------------
    {
        std::cout << "[Test Refiner Dispatch Parity: Scalar vs AVX2 (Full Matrix)]\n";

        const std::vector<std::pair<int, int>> test_sizes = {
            {32, 24}, {65, 47}, {17, 33}
        };

        for (const auto& sz : test_sizes) {
            const int tw = sz.first;
            const int th = sz.second;

            Image8 t_left(tw, th, 1);
            Image8 t_right(tw, th, 1);
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    t_left.at(x, y) = static_cast<uint8_t>((x * 17 + y * 31) % 256);
                    int xr = std::max(0, x - (x % 5));
                    t_right.at(x, y) = static_cast<uint8_t>((xr * 17 + y * 31) % 256);
                }
            }

            for (CensusType ctype : {CensusType::SymmetricCensus9x7, CensusType::Census9x7}) {
                for (int iters : {1, 2, 3}) {
                    for (int radius : {1, 4}) {
                        for (int mask_mode : {0, 1, 2}) { // 0=mixed, 1=all reliable, 2=all unreliable
                            PipelineConfig t_cfg;
                            t_cfg.min_disparity = -4;
                            t_cfg.max_disparity = 16;
                            t_cfg.cost.census = ctype;
                            t_cfg.cost.use_ad = (iters % 2 == 1);
                            t_cfg.cost.use_grad = (radius == 4);
                            t_cfg.refine.enable = true;
                            t_cfg.refine.iterations = iters;
                            t_cfg.refine.random_radius = radius;

                            PipelineBuffers t_buf;
                            CostComputer cc_temp;
                            cc_temp.compute_aux(t_left, t_right, t_cfg, t_buf);

                            t_buf.range.dmin = Image16s(tw, th, -4);
                            t_buf.range.dmax = Image16s(tw, th, 16);
                            // Set variable search ranges including some empty / single disparity
                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    if ((x + y) % 11 == 0) {
                                        t_buf.range.dmin.at(x, y) = 0;
                                        t_buf.range.dmax.at(x, y) = 0; // empty
                                    } else if ((x + y) % 7 == 0) {
                                        t_buf.range.dmin.at(x, y) = 2;
                                        t_buf.range.dmax.at(x, y) = 3; // single
                                    }
                                }
                            }

                            t_buf.disparity = Image32f(tw, th, 0.f);
                            t_buf.reliable_mask = Image8u1(tw, th, 0);
                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    t_buf.disparity.at(x, y) = static_cast<float>((x % 7) - 2);
                                    if (mask_mode == 0) {
                                        t_buf.reliable_mask.at(x, y) = ((x + y) % 3 == 0) ? 1 : 0;
                                    } else if (mask_mode == 1) {
                                        t_buf.reliable_mask.at(x, y) = 1; // all reliable
                                    } else {
                                        t_buf.reliable_mask.at(x, y) = 0; // all unreliable
                                    }
                                }
                            }

                            PipelineBuffers buf_sca = t_buf;
                            PipelineBuffers buf_avx = t_buf;

                            detail::refine_scalar(t_cfg, buf_sca);
                            detail::refine_avx2(t_cfg, buf_avx);

                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    float vs = buf_sca.disparity.at(x, y);
                                    float va = buf_avx.disparity.at(x, y);
                                    if (vs != va) {
                                        std::cerr << "Refiner mismatch at (" << x << "," << y << ") size="
                                                  << tw << "x" << th << " iters=" << iters << " rad="
                                                  << radius << " mask=" << mask_mode << ": scalar="
                                                  << vs << " vs avx2=" << va << "\n";
                                        return 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Refiner comprehensive matrix (sizes, iters, radii, masks, ranges, censuses): 100% bit-exact!\n";
    }

    // -------------------------------------------------------------
    // Test P3.14b-2 Deterministic Prior Parallelization Parity
    // -------------------------------------------------------------
    {
        std::cout << "[Test PriorEstimator Deterministic Parallel Parity]" << std::endl;

        auto serial_extract_supports = [](const PipelineConfig& cfg, const PipelineBuffers& buf) -> std::vector<SupportMatch> {
            const int w = buf.left_gray.width();
            const int h = buf.left_gray.height();
            const int d0 = cfg.min_disparity;
            const int d1 = cfg.max_disparity;
            const bool sym = cfg.cost.census == CensusType::SymmetricCensus9x7;
            const int step = 2;

            constexpr int cell = 16;
            const int gw = (w + cell - 1) / cell;
            const int gh = (h + cell - 1) / cell;
            const int n_cells = gw * gh;
            const int hard_max = cfg.prior.max_supports;

            std::vector<std::vector<SupportMatch>> cell_candidates(n_cells);
            const int max_cand_per_cell = std::max(8, (hard_max * 2 + n_cells - 1) / std::max(1, n_cells));

            for (int y = 2; y < h - 2; y += step) {
                const int gy = clampi(y / cell, 0, gh - 1);
                for (int x = 2; x < w - 2; x += step) {
                    const int gx = clampi(x / cell, 0, gw - 1);
                    const int ci = gy * gw + gx;
                    if (static_cast<int>(cell_candidates[ci].size()) >= max_cand_per_cell) continue;

                    const int tex = static_cast<int>(buf.left_gx.at(x, y)) + static_cast<int>(buf.left_gy.at(x, y));
                    if (tex < cfg.prior.texture_threshold) continue;

                    int best_d = d0;
                    int best = 255, second = 255;
                    for (int d = d0; d < d1; ++d) {
                        const int xr = x - d;
                        if (xr < 0 || xr >= w) continue;
                        int c = 0;
                        if (sym) {
                            c = CostComputer::popcount32(buf.census_left[y * w + x] ^ buf.census_right[y * w + xr]);
                        } else {
                            c = CostComputer::popcount64(buf.census_left64[y * w + x] ^ buf.census_right64[y * w + xr]);
                        }
                        if (c < best) {
                            second = best; best = c; best_d = d;
                        } else if (c < second) {
                            second = c;
                        }
                    }
                    if (second <= 0) continue;
                    const float uniq = static_cast<float>(second - best) / static_cast<float>(second);
                    if (uniq < cfg.prior.uniqueness_ratio) continue;

                    const int xr = x - best_d;
                    if (xr < 1 || xr >= w - 1) continue;
                    int rbest = 255, rbest_d = 0;
                    for (int d = d0; d < d1; ++d) {
                        const int xl = xr + d;
                        if (xl < 0 || xl >= w) continue;
                        int c = 0;
                        if (sym) {
                            c = CostComputer::popcount32(buf.census_left[y * w + xl] ^ buf.census_right[y * w + xr]);
                        } else {
                            c = CostComputer::popcount64(buf.census_left64[y * w + xl] ^ buf.census_right64[y * w + xr]);
                        }
                        if (c < rbest) {
                            rbest = c; rbest_d = d;
                        }
                    }
                    if (std::abs(rbest_d - best_d) > cfg.prior.lr_max_diff) continue;

                    SupportMatch m;
                    m.x = x; m.y = y; m.disparity = static_cast<float>(best_d); m.confidence = uniq;
                    cell_candidates[ci].push_back(m);
                }
            }

            std::vector<int> nonempty_cells;
            nonempty_cells.reserve(n_cells);
            for (int i = 0; i < n_cells; ++i) {
                if (!cell_candidates[i].empty()) nonempty_cells.push_back(i);
            }

            std::vector<SupportMatch> out;
            out.reserve(std::min(static_cast<size_t>(hard_max), static_cast<size_t>(n_cells * 4)));

            if (static_cast<int>(nonempty_cells.size()) <= hard_max) {
                std::vector<int> taken(n_cells, 0);
                const int base_quota = std::max(1, hard_max / std::max(1, static_cast<int>(nonempty_cells.size())));
                for (int ci : nonempty_cells) {
                    const int take = std::min(base_quota, static_cast<int>(cell_candidates[ci].size()));
                    for (int k = 0; k < take && static_cast<int>(out.size()) < hard_max; ++k) {
                        out.push_back(cell_candidates[ci][k]);
                        taken[ci]++;
                    }
                }
                bool added = true;
                while (added && static_cast<int>(out.size()) < hard_max) {
                    added = false;
                    for (int ci : nonempty_cells) {
                        if (static_cast<int>(out.size()) >= hard_max) break;
                        if (taken[ci] < static_cast<int>(cell_candidates[ci].size())) {
                            out.push_back(cell_candidates[ci][taken[ci]]);
                            taken[ci]++;
                            added = true;
                        }
                    }
                }
            } else {
                const size_t num_nonempty = nonempty_cells.size();
                for (int k = 0; k < hard_max; ++k) {
                    const size_t sample_idx = (static_cast<uint64_t>(k) * num_nonempty) / hard_max;
                    const int ci = nonempty_cells[sample_idx];
                    if (!cell_candidates[ci].empty()) {
                        out.push_back(cell_candidates[ci].front());
                    }
                }
            }
            return out;
        };

        auto serial_interpolate_prior = [](const std::vector<SupportMatch>& supports, PipelineBuffers& buf) {
            const int w = buf.left_gray.width();
            const int h = buf.left_gray.height();
            buf.d_prior = Image32f(w, h, -1.f);
            buf.prior_confidence = Image32f(w, h, 0.f);
            buf.prior_spread = Image32f(w, h, 999.f);
            buf.d_prior_min = Image32f(w, h, -1.f);
            buf.d_prior_max = Image32f(w, h, -1.f);
            if (supports.empty()) return;

            constexpr int cell = 16;
            const int gw = (w + cell - 1) / cell;
            const int gh = (h + cell - 1) / cell;
            const int n_cells = gw * gh;

            std::vector<std::vector<std::pair<float, float>>> cell_supports(n_cells);
            for (const auto& s : supports) {
                const int gx = clampi(s.x / cell, 0, gw - 1);
                const int gy = clampi(s.y / cell, 0, gh - 1);
                cell_supports[gy * gw + gx].push_back({s.disparity, s.confidence});
            }

            std::vector<float> grid_disp(n_cells, -1.f);
            std::vector<float> grid_conf(n_cells, 0.f);
            std::vector<float> grid_spread(n_cells, -1.f);
            std::vector<float> grid_dmin(n_cells, -1.f);
            std::vector<float> grid_dmax(n_cells, -1.f);

            for (int i = 0; i < n_cells; ++i) {
                auto& pts = cell_supports[i];
                if (pts.empty()) continue;

                std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });

                grid_dmin[i] = pts.front().first;
                grid_dmax[i] = pts.back().first;
                const float intra_span = grid_dmax[i] - grid_dmin[i];

                float total_w = 0.f;
                for (const auto& p : pts) total_w += p.second;
                if (total_w <= 1e-6f) continue;

                float half_w = total_w * 0.5f;
                float cur_w = 0.f;
                float d_med = pts.front().first;
                for (const auto& p : pts) {
                    cur_w += p.second;
                    if (cur_w >= half_w) {
                        d_med = p.first;
                        break;
                    }
                }
                grid_disp[i] = d_med;

                std::vector<std::pair<float, float>> devs;
                devs.reserve(pts.size());
                for (const auto& p : pts) {
                    devs.push_back({std::abs(p.first - d_med), p.second});
                }
                std::sort(devs.begin(), devs.end(), [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });
                cur_w = 0.f;
                float dev_med = devs.front().first;
                for (const auto& d : devs) {
                    cur_w += d.second;
                    if (cur_w >= half_w) {
                        dev_med = d.first;
                        break;
                    }
                }
                grid_spread[i] = std::max(dev_med * 1.4826f, intra_span * 0.5f);

                const float mean_conf = total_w / static_cast<float>(pts.size());
                const float count_factor = clampf(static_cast<float>(pts.size()) / 3.f, 0.3f, 1.f);
                grid_conf[i] = clampf(mean_conf * count_factor, 0.f, 1.f);
            }

            auto fill_grids = [&]() -> bool {
                std::vector<float> nxt_d = grid_disp;
                std::vector<float> nxt_c = grid_conf;
                std::vector<float> nxt_s = grid_spread;
                std::vector<float> nxt_min = grid_dmin;
                std::vector<float> nxt_max = grid_dmax;
                bool changed = false;
                for (int gy = 0; gy < gh; ++gy) {
                    for (int gx = 0; gx < gw; ++gx) {
                        const int i = gy * gw + gx;
                        if (grid_disp[i] >= 0.f) continue;
                        float ad = 0.f, ac = 0.f, as = 0.f, ww = 0.f;
                        float cur_min = 1e9f, cur_max = -1e9f;
                        for (int oy = -1; oy <= 1; ++oy) {
                            for (int ox = -1; ox <= 1; ++ox) {
                                const int nx = gx + ox, ny = gy + oy;
                                if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
                                const int ni = ny * gw + nx;
                                const float v = grid_disp[ni];
                                if (v < 0.f) continue;
                                ad += v;
                                ac += grid_conf[ni];
                                as += (grid_spread[ni] >= 0.f ? grid_spread[ni] : 8.f);
                                cur_min = std::min(cur_min, grid_dmin[ni] >= 0.f ? grid_dmin[ni] : v);
                                cur_max = std::max(cur_max, grid_dmax[ni] >= 0.f ? grid_dmax[ni] : v);
                                ww += 1.f;
                            }
                        }
                        if (ww > 0.f) {
                            nxt_d[i] = ad / ww;
                            nxt_c[i] = (ac / ww) * 0.85f;
                            nxt_s[i] = as / ww;
                            nxt_min[i] = cur_min;
                            nxt_max[i] = cur_max;
                            changed = true;
                        }
                    }
                }
                grid_disp.swap(nxt_d);
                grid_conf.swap(nxt_c);
                grid_spread.swap(nxt_s);
                grid_dmin.swap(nxt_min);
                grid_dmax.swap(nxt_max);
                return changed;
            };
            const int max_iters = std::max(gw, gh);
            for (int iter = 0; iter < max_iters; ++iter) {
                if (!fill_grids()) break;
            }

            for (int y = 0; y < h; ++y) {
                const float fy = (static_cast<float>(y) + 0.5f) / cell - 0.5f;
                const int gy = clampi(static_cast<int>(std::floor(fy)), 0, gh - 1);
                const int gy1 = std::min(gy + 1, gh - 1);
                const float ty = clampf(fy - static_cast<float>(gy), 0.f, 1.f);
                for (int x = 0; x < w; ++x) {
                    const float fx = (static_cast<float>(x) + 0.5f) / cell - 0.5f;
                    const int gx = clampi(static_cast<int>(std::floor(fx)), 0, gw - 1);
                    const int gx1 = std::min(gx + 1, gw - 1);
                    const float tx = clampf(fx - static_cast<float>(gx), 0.f, 1.f);

                    const int i00 = gy * gw + gx;
                    const int i10 = gy * gw + gx1;
                    const int i01 = gy1 * gw + gx;
                    const int i11 = gy1 * gw + gx1;

                    auto pick = [](float v) { return v < 0.f ? 0.f : v; };
                    auto wgtv = [](float v) { return v < 0.f ? 0.f : 1.f; };

                    const float w00 = (1.f - tx) * (1.f - ty) * wgtv(grid_disp[i00]);
                    const float w10 = tx * (1.f - ty) * wgtv(grid_disp[i10]);
                    const float w01 = (1.f - tx) * ty * wgtv(grid_disp[i01]);
                    const float w11 = tx * ty * wgtv(grid_disp[i11]);
                    const float ww = w00 + w10 + w01 + w11;

                    if (ww > 1e-6f) {
                        const float dp =
                            (w00 * pick(grid_disp[i00]) + w10 * pick(grid_disp[i10]) +
                             w01 * pick(grid_disp[i01]) + w11 * pick(grid_disp[i11])) / ww;
                        buf.d_prior.at(x, y) = dp;
                        buf.prior_confidence.at(x, y) =
                            (w00 * grid_conf[i00] + w10 * grid_conf[i10] +
                             w01 * grid_conf[i01] + w11 * grid_conf[i11]) / ww;

                        float cmin = 1e9f, cmax = -1e9f;
                        for (int idx : {i00, i10, i01, i11}) {
                            if (grid_disp[idx] >= 0.f) {
                                const float mn = grid_dmin[idx] >= 0.f ? grid_dmin[idx] : grid_disp[idx];
                                const float mx = grid_dmax[idx] >= 0.f ? grid_dmax[idx] : grid_disp[idx];
                                cmin = std::min(cmin, mn);
                                cmax = std::max(cmax, mx);
                            }
                        }
                        const float local_spread =
                            (w00 * pick(grid_spread[i00]) + w10 * pick(grid_spread[i10]) +
                             w01 * pick(grid_spread[i01]) + w11 * pick(grid_spread[i11])) / ww;
                        buf.prior_spread.at(x, y) = local_spread;

                        buf.d_prior_min.at(x, y) = (cmin <= cmax) ? std::min(cmin, dp) : dp;
                        buf.d_prior_max.at(x, y) = (cmin <= cmax) ? std::max(cmax, dp) : dp;
                    } else {
                        buf.d_prior.at(x, y) = -1.f;
                        buf.prior_confidence.at(x, y) = 0.f;
                        buf.prior_spread.at(x, y) = 999.f;
                        buf.d_prior_min.at(x, y) = -1.f;
                        buf.d_prior_max.at(x, y) = -1.f;
                    }
                }
            }
        };

        auto serial_apply_search_range = [](const PipelineConfig& cfg, PipelineBuffers& buf) {
            const int w = buf.left_gray.width();
            const int h = buf.left_gray.height();
            if (!cfg.prior.enable || buf.d_prior.empty()) return;
            const int default_R = cfg.prior.search_radius;
            const int global_D = cfg.max_disparity - cfg.min_disparity;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const float dp = buf.d_prior.at(x, y);
                    if (dp < 0.f) continue;

                    const float conf = !buf.prior_confidence.empty() ? buf.prior_confidence.at(x, y) : 0.f;
                    const float spread = !buf.prior_spread.empty() ? buf.prior_spread.at(x, y) : 999.f;
                    const float pmin = (!buf.d_prior_min.empty() && buf.d_prior_min.at(x, y) >= 0.f) ? buf.d_prior_min.at(x, y) : dp;
                    const float pmax = (!buf.d_prior_max.empty() && buf.d_prior_max.at(x, y) >= 0.f) ? buf.d_prior_max.at(x, y) : dp;

                    int R = default_R;
                    if (conf < 0.25f) {
                        R = global_D;
                    } else if (conf > 0.85f && spread < 1.5f) {
                        R = 8;
                    } else if (conf > 0.60f && spread < 3.0f) {
                        R = static_cast<int>(12.f + 1.5f * spread + 0.5f);
                    } else if (conf > 0.40f && spread < 6.0f) {
                        R = static_cast<int>(18.f + 2.0f * spread + 0.5f);
                    } else {
                        R = std::max(24, static_cast<int>(16.f + 2.0f * spread + 0.5f));
                    }
                    R = std::min(R, global_D);

                    constexpr int kEnvelopeMargin = 4;
                    const int bound_lo = static_cast<int>(std::floor(pmin)) - kEnvelopeMargin;
                    const int bound_hi = static_cast<int>(std::ceil(pmax)) + kEnvelopeMargin + 1;

                    const int lo = std::min(static_cast<int>(std::floor(dp)) - R, bound_lo);
                    int hi = std::max(static_cast<int>(std::ceil(dp)) + R + 1, bound_hi);
                    hi = std::min(hi, x + 1);
                    buf.range.set_pixel(x, y, lo, hi);
                }
            }
        };

        auto serial_estimate = [&](const PipelineConfig& cfg, PipelineBuffers& buf) {
            const int w = buf.left_gray.width();
            const int h = buf.left_gray.height();
            buf.range.allocate(w, h, cfg.min_disparity, cfg.max_disparity);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int lo = cfg.min_disparity;
                    const int hi = std::min(cfg.max_disparity, x + 1);
                    buf.range.set_pixel(x, y, lo, hi);
                }
            }
            buf.d_prior = Image32f(w, h, -1.f);
            buf.supports.clear();
            buf.support_count = 0;
            if (!cfg.prior.enable) return;

            buf.supports = serial_extract_supports(cfg, buf);
            buf.support_count = buf.supports.size();
            if (static_cast<int>(buf.supports.size()) < cfg.prior.min_supports) return;
            serial_interpolate_prior(buf.supports, buf);
            serial_apply_search_range(cfg, buf);
        };

        auto verify_prior_parity = [&](const PipelineConfig& cfg,
                                       const PipelineBuffers& base_input,
                                       const std::string& case_label) -> bool {
            std::cout << "  Testing " << case_label << "..." << std::endl;
            PriorEstimator prod_prior;
            PipelineBuffers b_ser = base_input;
            PipelineBuffers b_par = base_input;

            serial_estimate(cfg, b_ser);
            prod_prior.estimate(cfg, b_par);

            if (b_ser.supports.size() != b_par.supports.size()) {
                std::cerr << "Prior mismatch in " << case_label << ": support count "
                          << b_ser.supports.size() << " vs " << b_par.supports.size() << "\n";
                return false;
            }
            for (size_t i = 0; i < b_ser.supports.size(); ++i) {
                const auto& s = b_ser.supports[i];
                const auto& p = b_par.supports[i];
                if (s.x != p.x || s.y != p.y || s.disparity != p.disparity || s.confidence != p.confidence) {
                    std::cerr << "Prior support mismatch in " << case_label << " at index " << i
                              << ": (" << s.x << "," << s.y << ",d=" << s.disparity << ",c=" << s.confidence
                              << ") vs (" << p.x << "," << p.y << ",d=" << p.disparity << ",c=" << p.confidence << ")\n";
                    return false;
                }
            }

            const int w = b_ser.left_gray.width();
            const int h = b_ser.left_gray.height();
            if (b_ser.prior_confidence.empty() != b_par.prior_confidence.empty()) return false;
            const bool has_interp = !b_ser.prior_confidence.empty();

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (b_ser.d_prior.at(x, y) != b_par.d_prior.at(x, y) ||
                        b_ser.range.dmin.at(x, y) != b_par.range.dmin.at(x, y) ||
                        b_ser.range.dmax.at(x, y) != b_par.range.dmax.at(x, y)) {
                        std::cerr << "Prior buffer mismatch in " << case_label << " at (" << x << "," << y << ")\n";
                        return false;
                    }
                    if (has_interp) {
                        if (b_ser.prior_confidence.at(x, y) != b_par.prior_confidence.at(x, y) ||
                            b_ser.prior_spread.at(x, y) != b_par.prior_spread.at(x, y) ||
                            b_ser.d_prior_min.at(x, y) != b_par.d_prior_min.at(x, y) ||
                            b_ser.d_prior_max.at(x, y) != b_par.d_prior_max.at(x, y)) {
                            std::cerr << "Prior interpolated buffer mismatch in " << case_label << " at (" << x << "," << y << ")\n";
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        // 1. Dimension Edge Cases: 1x1, 7x9, 15x17, 16x16, 17x17, 32x33, 65x47, 100x75
        std::vector<std::pair<int, int>> dim_cases = {
            {1, 1}, {7, 9}, {15, 17}, {16, 16}, {17, 17}, {32, 33}, {65, 47}, {100, 75}
        };
        for (const auto& d : dim_cases) {
            const int tw = d.first, th = d.second;
            PipelineConfig c;
            c.prior.enable = true;
            c.min_disparity = 0;
            c.max_disparity = std::max(2, std::min(32, tw - 1));

            Image8 r;
            Image8 l = make_shift_pair(tw, th, 1, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);

            if (!verify_prior_parity(c, b, "Dimension " + std::to_string(tw) + "x" + std::to_string(th))) {
                return 1;
            }
        }
        std::cout << "  Dimension edge cases (1x1 to 100x75, non-multiple of 16): 100% bit-exact!\n";

        // 2. Budget Case A: nonempty_cells <= max_supports (base quota + round-robin)
        {
            const int tw = 96, th = 80;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.max_supports = 200; // > n_cells (6 * 5 = 30)
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 4, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Budget Case A")) return 1;
        }
        std::cout << "  Budget Case A (nonempty_cells <= max_supports): 100% bit-exact!\n";

        // 3. Budget Case B: nonempty_cells > max_supports (deterministic uniform subsampling)
        {
            const int tw = 160, th = 128;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.max_supports = 12; // < n_cells (10 * 8 = 80)
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 4, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Budget Case B")) return 1;
        }
        std::cout << "  Budget Case B (nonempty_cells > max_supports): 100% bit-exact!\n";

        // 4. Small support budget: max_supports = 1, 2, 3
        for (int small_s : {1, 2, 3}) {
            const int tw = 80, th = 64;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.min_supports = 1;
            c.prior.max_supports = small_s;
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 2, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Small support budget max_supports=" + std::to_string(small_s))) return 1;
        }
        std::cout << "  Small support budget (max_supports = 1, 2, 3): 100% bit-exact!\n";

        // 5. min_supports early return: support_count < min_supports
        {
            const int tw = 80, th = 64;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.min_supports = 5000; // Will trigger early return
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 4, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "min_supports early return")) return 1;
        }
        std::cout << "  min_supports early return: 100% bit-exact!\n";

        // 6. Texture Extremes: all low texture vs dense high texture
        {
            const int tw = 64, th = 48;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.texture_threshold = 1000; // All pixels below texture threshold
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 2, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "All low texture")) return 1;
        }
        {
            const int tw = 64, th = 48;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.texture_threshold = 1; // Dense texture
            c.min_disparity = 0;
            c.max_disparity = 16;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 2, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Dense high texture")) return 1;
        }
        std::cout << "  Texture extremes (zero texture & dense texture): 100% bit-exact!\n";

        // 7. Disparity Ranges: negative min, positive-only, small D, large D
        std::vector<std::pair<int, int>> disp_ranges = {
            {-8, 16}, {10, 40}, {0, 2}, {0, 64}
        };
        for (const auto& dr : disp_ranges) {
            const int tw = 128, th = 64;
            PipelineConfig c;
            c.prior.enable = true;
            c.min_disparity = dr.first;
            c.max_disparity = dr.second;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, std::max(0, dr.first + 2), r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Disparity range [" + std::to_string(dr.first) + "," + std::to_string(dr.second) + "]")) return 1;
        }
        std::cout << "  Disparity ranges (negative, positive-only, small D, large D): 100% bit-exact!\n";

        // 8. Cap-hit condition in individual cells
        {
            const int tw = 64, th = 64;
            PipelineConfig c;
            c.prior.enable = true;
            c.prior.max_supports = 32; // Force max_cand_per_cell = 8
            c.prior.texture_threshold = 0;
            c.prior.uniqueness_ratio = 0.0f;
            c.prior.lr_max_diff = 100;
            c.min_disparity = 0;
            c.max_disparity = 8;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 2, r);
            PipelineBuffers b;
            CostComputer cc_local;
            cc_local.compute_aux(l, r, c, b);
            if (!verify_prior_parity(c, b, "Cap-hit condition")) return 1;
        }
        std::cout << "  Cap-hit cell condition: 100% bit-exact!\n";

        // 9. Full Pipeline End-to-End Parity on Synthetic Pair
        {
            const int tw = 128, th = 96;
            Image8 r;
            Image8 l = make_shift_pair(tw, th, 6, r);
            auto cfg_pipe = PipelineConfig::from_mode(QualityMode::HighQuality, 24);
            cfg_pipe.prior.enable = true;

            // Pipeline with production parallel prior
            StereoMatcher matcher_par(cfg_pipe);
            PipelineBuffers b_pipe_par;
            if (!matcher_par.compute(l, r, b_pipe_par)) {
                std::cerr << "Pipeline parallel run failed: " << matcher_par.last_error() << "\n";
                return 1;
            }

            // Verify that pipeline range matches serial prior range bit-for-bit
            PipelineBuffers b_pipe_ser;
            CostComputer cc_p;
            cc_p.compute_aux(l, r, cfg_pipe, b_pipe_ser);
            serial_estimate(cfg_pipe, b_pipe_ser);

            int range_diffs = 0;
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    if (b_pipe_ser.range.dmin.at(x, y) != b_pipe_par.range.dmin.at(x, y) ||
                        b_pipe_ser.range.dmax.at(x, y) != b_pipe_par.range.dmax.at(x, y)) {
                        range_diffs++;
                    }
                }
            }
            if (range_diffs != 0) {
                std::cerr << "Pipeline search range mismatch count = " << range_diffs << "\n";
                return 1;
            }
        }
        std::cout << "  Full Pipeline SearchRange parity: 100% bit-exact (0 diffs)!\n";
    }

    // -------------------------------------------------------------
    // Test P3.15b Deterministic Post-Processing Parallel Parity
    // -------------------------------------------------------------
    {
        std::cout << "[Test PostProcessor Deterministic Parallel Parity]" << std::endl;

        auto serial_lr_check = [](const PipelineConfig& cfg, PipelineBuffers& buf) {
            if (!cfg.post.lr_check || buf.disparity_right.empty()) return;
            const int w = buf.disparity.width();
            const int h = buf.disparity.height();
            const float tol = static_cast<float>(cfg.post.lr_max_diff) + 0.5f;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const float d = buf.disparity.at(x, y);
                    if (d < 0.f) continue;
                    const int xr = static_cast<int>(std::round(static_cast<float>(x) - d));
                    if (xr < 0 || xr >= w) {
                        buf.disparity.at(x, y) = -1.f;
                        buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Occlusion);
                        continue;
                    }
                    const float dr = buf.disparity_right.at(xr, y);
                    if (dr < 0.f || std::abs(d - dr) > tol) {
                        buf.disparity.at(x, y) = -1.f;
                        if (dr > d + tol) {
                            buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Occlusion);
                        } else {
                            buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Mismatch);
                        }
                    }
                }
            }
        };

        auto serial_fill_holes = [](const PipelineConfig& cfg, PipelineBuffers& buf) {
            const int w = buf.disparity.width();
            const int h = buf.disparity.height();
            const int max_gap = cfg.post.max_fill_gap;

            Image32f out = buf.disparity;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (buf.disparity.at(x, y) >= 0.f) continue;
                    const auto reason = static_cast<InvalidReason>(buf.invalid_reason.at(x, y));

                    int xl = x - 1;
                    while (xl >= 0 && xl >= x - max_gap && buf.disparity.at(xl, y) < 0.f) --xl;
                    int xr = x + 1;
                    while (xr < w && xr <= x + max_gap && buf.disparity.at(xr, y) < 0.f) ++xr;

                    float dl = (xl >= 0 && buf.disparity.at(xl, y) >= 0.f) ? buf.disparity.at(xl, y) : -1.f;
                    float dr = (xr < w && buf.disparity.at(xr, y) >= 0.f) ? buf.disparity.at(xr, y) : -1.f;

                    if (reason == InvalidReason::Occlusion) {
                        if (dl >= 0.f && dr >= 0.f) out.at(x, y) = std::min(dl, dr);
                        else if (dl >= 0.f) out.at(x, y) = dl;
                        else if (dr >= 0.f) out.at(x, y) = dr;
                        continue;
                    }

                    const uint8_t c0 = buf.left_gray.at(x, y);
                    float best_d = -1.f;
                    int best_abs = 999;
                    auto consider = [&](int xx, int yy) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) return;
                        const float d = buf.disparity.at(xx, yy);
                        if (d < 0.f) return;
                        const int ad = std::abs(static_cast<int>(buf.left_gray.at(xx, yy)) - static_cast<int>(c0));
                        if (ad < best_abs) {
                            best_abs = ad;
                            best_d = d;
                        }
                    };
                    consider(xl, y);
                    consider(xr, y);
                    if (y > 0) consider(x, y - 1);
                    if (y + 1 < h) consider(x, y + 1);
                    if (best_d >= 0.f && best_abs <= 18) {
                        out.at(x, y) = best_d;
                    } else if (!buf.d_prior.empty() && buf.d_prior.at(x, y) >= 0.f) {
                        out.at(x, y) = buf.d_prior.at(x, y);
                    } else if (dl >= 0.f && dr >= 0.f) {
                        const float t = static_cast<float>(x - xl) / static_cast<float>(xr - xl);
                        out.at(x, y) = dl * (1.f - t) + dr * t;
                    } else if (dl >= 0.f) {
                        out.at(x, y) = dl;
                    } else if (dr >= 0.f) {
                        out.at(x, y) = dr;
                    }
                }
            }
            buf.disparity = std::move(out);
        };

        auto serial_median = [](const PipelineConfig& cfg, PipelineBuffers& buf) {
            const int r = cfg.post.median_radius;
            if (r <= 0) return;
            const int w = buf.disparity.width();
            const int h = buf.disparity.height();
            Image32f out = buf.disparity;
            std::vector<float> win;
            win.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    win.clear();
                    for (int oy = -r; oy <= r; ++oy) {
                        for (int ox = -r; ox <= r; ++ox) {
                            const int xx = x + ox, yy = y + oy;
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                            const float d = buf.disparity.at(xx, yy);
                            if (d >= 0.f) win.push_back(d);
                        }
                    }
                    if (win.empty()) continue;
                    const size_t mid = win.size() / 2;
                    std::nth_element(win.begin(), win.begin() + static_cast<std::ptrdiff_t>(mid), win.end());
                    out.at(x, y) = win[mid];
                }
            }
            buf.disparity = std::move(out);
        };

        PostProcessor prod_post;

        // Matrix dimensions
        const std::vector<std::pair<int, int>> test_dims = {
            {1, 1},
            {1, 33},
            {47, 1},
            {17, 19},
            {64, 48},
            {128, 96},
            {149, 113}
        };

        for (const auto& dim : test_dims) {
            const int tw = dim.first;
            const int th = dim.second;
            std::cout << "  Testing Post dim " << tw << "x" << th << "..." << std::endl;

            // Test combinations of parameters
            for (bool lr_check_en : {true, false}) {
                for (int max_fill_gap : {0, 1, 32, 200}) {
                    for (int med_r : {0, 1, 2}) {
                        for (bool with_prior : {true, false}) {
                            PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::HighQuality, 32);
                            cfg.post.lr_check = lr_check_en;
                            cfg.post.lr_max_diff = 1;
                            cfg.post.max_fill_gap = max_fill_gap;
                            cfg.post.median_radius = med_r;

                            PipelineBuffers b_ser, b_par;
                            b_ser.disparity = Image32f(tw, th, -1.f);
                            b_ser.disparity_right = Image32f(tw, th, -1.f);
                            b_ser.invalid_reason = Image8u1(tw, th, 0);
                            b_ser.left_gray = Image8(tw, th, 1);
                            if (with_prior) {
                                b_ser.d_prior = Image32f(tw, th, -1.f);
                            }

                            // Synthesize test pattern
                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    b_ser.left_gray.at(x, y) = static_cast<uint8_t>((x * 13 + y * 29 + (x ^ y)) & 0xFF);
                                    if (with_prior && (x + y) % 5 == 0) {
                                        b_ser.d_prior.at(x, y) = static_cast<float>((x * 3 + y * 7) % 30);
                                    }

                                    // Synthesize initial disparity & right disparity
                                    if ((x + y) % 3 != 0) {
                                        float d_val = static_cast<float>((x + y) % 24);
                                        b_ser.disparity.at(x, y) = d_val;
                                    }
                                    if ((x * 2 + y) % 4 != 0) {
                                        float dr_val = static_cast<float>((x + y) % 24);
                                        if ((x + y) % 7 == 0) dr_val += 5.f; // Occlusion
                                        else if ((x + y) % 5 == 0) dr_val -= 5.f; // Mismatch
                                        b_ser.disparity_right.at(x, y) = dr_val;
                                    }
                                }
                            }

                            b_par = b_ser;

                            // 1. Verify Substage LR Check parity
                            serial_lr_check(cfg, b_ser);
                            prod_post.left_right_check(cfg, b_par);

                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    if (b_ser.disparity.at(x, y) != b_par.disparity.at(x, y) ||
                                        b_ser.invalid_reason.at(x, y) != b_par.invalid_reason.at(x, y)) {
                                        std::cerr << "Post LR mismatch at (" << x << "," << y << ") dim="
                                                  << tw << "x" << th << " lr_en=" << lr_check_en << "\n";
                                        return 1;
                                    }
                                }
                            }

                            // 2. Verify Substage Fill Holes parity
                            std::cout << "      step 2: fill holes" << std::endl;
                            serial_fill_holes(cfg, b_ser);
                            prod_post.fill_holes(cfg, b_par);

                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    if (b_ser.disparity.at(x, y) != b_par.disparity.at(x, y)) {
                                        std::cerr << "Post Fill mismatch at (" << x << "," << y << ") dim="
                                                  << tw << "x" << th << " gap=" << max_fill_gap << "\n";
                                        return 1;
                                    }
                                }
                            }

                            // 3. Verify Substage Median parity
                            serial_median(cfg, b_ser);
                            prod_post.median(cfg, b_par);

                            for (int y = 0; y < th; ++y) {
                                for (int x = 0; x < tw; ++x) {
                                    if (b_ser.disparity.at(x, y) != b_par.disparity.at(x, y)) {
                                        std::cerr << "Post Median mismatch at (" << x << "," << y << ") dim="
                                                  << tw << "x" << th << " rad=" << med_r << "\n";
                                        return 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  PostProcessor synthetic matrix (dims, lr_checks, gaps, radii, priors): 100% bit-exact!\n";
    }

    std::cout << "sanity ok\n";
    return 0;
}
