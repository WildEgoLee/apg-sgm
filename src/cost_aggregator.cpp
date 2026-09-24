#include "apg_sgm/cost_aggregator.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg {

void CostAggregator::build_cross_arms(const Image8& gray, const AggregationParams& p,
                                      std::vector<int16_t>& left, std::vector<int16_t>& right,
                                      std::vector<int16_t>& up, std::vector<int16_t>& down) const {
    const int w = gray.width();
    const int h = gray.height();
    const int n = w * h;
    left.assign(n, 0);
    right.assign(n, 0);
    up.assign(n, 0);
    down.assign(n, 0);
    const int Lmax = std::max(1, p.max_arm_length);
    const int tau = p.color_threshold;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t c0 = gray.at(x, y);
            int l = 0;
            while (l < Lmax && x - l - 1 >= 0 &&
                   std::abs(static_cast<int>(gray.at(x - l - 1, y)) - static_cast<int>(c0)) <= tau) {
                ++l;
            }
            int r = 0;
            while (r < Lmax && x + r + 1 < w &&
                   std::abs(static_cast<int>(gray.at(x + r + 1, y)) - static_cast<int>(c0)) <= tau) {
                ++r;
            }
            int u = 0;
            while (u < Lmax && y - u - 1 >= 0 &&
                   std::abs(static_cast<int>(gray.at(x, y - u - 1)) - static_cast<int>(c0)) <= tau) {
                ++u;
            }
            int v = 0;
            while (v < Lmax && y + v + 1 < h &&
                   std::abs(static_cast<int>(gray.at(x, y + v + 1)) - static_cast<int>(c0)) <= tau) {
                ++v;
            }
            const int i = y * w + x;
            left[i] = static_cast<int16_t>(l);
            right[i] = static_cast<int16_t>(r);
            up[i] = static_cast<int16_t>(u);
            down[i] = static_cast<int16_t>(v);
        }
    }
}

void CostAggregator::aggregate_hv(PipelineBuffers& buf,
                                  const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                                  const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm) const {
    const int w = buf.cost.width();
    const int h = buf.cost.height();
    const int D = buf.cost.D();
    const int d0 = buf.cost.d0();
    CostVolume tmp(w, h, d0, d0 + D, kInvalidCost);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const int x0 = x - Larm[i];
            const int x1 = x + Rarm[i];
            const int lo_c = buf.range.dmin.at(x, y);
            const int hi_c = buf.range.dmax.at(x, y);
            uint16_t* dst = tmp.slice(x, y);
            for (int di = 0; di < D; ++di) {
                const int d = d0 + di;
                if (d < lo_c || d >= hi_c) {
                    dst[di] = kInvalidCost;
                    continue;
                }
                int acc = 0;
                int count = 0;
                for (int xx = x0; xx <= x1; ++xx) {
                    if (d < buf.range.dmin.at(xx, y) || d >= buf.range.dmax.at(xx, y)) {
                        continue;
                    }
                    const uint16_t c = buf.cost.slice(xx, y)[di];
                    if (c == kInvalidCost) continue;
                    acc += c;
                    ++count;
                }
                dst[di] = (count > 0) ? static_cast<uint16_t>(acc / count) : kInvalidCost;
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const int y0 = y - Uarm[i];
            const int y1 = y + Darm[i];
            const int lo_c = buf.range.dmin.at(x, y);
            const int hi_c = buf.range.dmax.at(x, y);
            uint16_t* dst = buf.cost.slice(x, y);
            for (int di = 0; di < D; ++di) {
                const int d = d0 + di;
                if (d < lo_c || d >= hi_c) {
                    dst[di] = kInvalidCost;
                    continue;
                }
                int acc = 0;
                int count = 0;
                for (int yy = y0; yy <= y1; ++yy) {
                    if (d < buf.range.dmin.at(x, yy) || d >= buf.range.dmax.at(x, yy)) {
                        continue;
                    }
                    const uint16_t c = tmp.slice(x, yy)[di];
                    if (c == kInvalidCost) continue;
                    acc += c;
                    ++count;
                }
                dst[di] = (count > 0) ? static_cast<uint16_t>(acc / count) : kInvalidCost;
            }
        }
    }
}

void CostAggregator::aggregate(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    if (!cfg.aggregation.enable) return;
    std::vector<int16_t> L, R, U, D;
    build_cross_arms(buf.left_gray, cfg.aggregation, L, R, U, D);
    const int iters = std::max(1, cfg.aggregation.iterations);
    for (int i = 0; i < iters; ++i) {
        aggregate_hv(buf, L, R, U, D);
    }
}

std::size_t CostAggregator::aggregate_hv_packed_streaming(
    PackedCostVolume16& packed_cost,
    const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
    const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm,
    int iterations) const {
    const int w = packed_cost.width();
    const int h = packed_cost.height();
    if (w <= 0 || h <= 0 || packed_cost.empty()) return 0;

    const auto layout = packed_cost.layout();
    if (!layout) return 0;

    int max_up = 0;
    int max_down = 0;
    for (int16_t v : Uarm) max_up = std::max(max_up, static_cast<int>(v));
    for (int16_t v : Darm) max_down = std::max(max_down, static_cast<int>(v));

    // Process horizontal rows in blocks to amortize OpenMP barriers.  Batching
    // needs B-1 extra ring slots versus the one-row pipeline: before vertical
    // rows from the current block are emitted, every horizontal row in that
    // block must coexist with the oldest still-needed row.
    constexpr int kBatchRows = 16;
    const int batch_rows = std::min(h, kBatchRows);
    const int ring_rows = std::max(
        1, std::min(h, max_up + max_down + batch_rows));

    struct HorizontalRow {
        std::vector<uint16_t> data;
    };

    struct NeighborInfo {
        const uint16_t* slice;
        int dmin;
        int dmax;
    };

    std::vector<HorizontalRow> ring(static_cast<std::size_t>(ring_rows));
    std::vector<std::size_t> slot_capacity(static_cast<std::size_t>(ring_rows), 0);

    auto row_begin = [&](int y) -> uint32_t {
        return layout->offsets[static_cast<std::size_t>(y) * w];
    };
    auto row_end = [&](int y) -> uint32_t {
        return layout->offsets[static_cast<std::size_t>(y + 1) * w];
    };

    // Pre-size every ring slot to the largest packed row that can map to it.
    // The slot then needs no allocation, resize or metadata update while the
    // OpenMP workers stream blocks through it.
    for (int y = 0; y < h; ++y) {
        const std::size_t states =
            static_cast<std::size_t>(row_end(y) - row_begin(y));
        const std::size_t slot = static_cast<std::size_t>(y % ring_rows);
        slot_capacity[slot] = std::max(slot_capacity[slot], states);
    }

    std::size_t workspace_bytes = 0;
    for (int s = 0; s < ring_rows; ++s) {
        auto& row = ring[static_cast<std::size_t>(s)];
        row.data.resize(slot_capacity[static_cast<std::size_t>(s)]);
        workspace_bytes += row.data.capacity() * sizeof(uint16_t);
    }

    auto horizontal_pixel = [&](int x, int y) {
        auto& row = ring[static_cast<std::size_t>(y % ring_rows)];

        const int lo = packed_cost.dmin(x, y);
        const int hi = packed_cost.dmax(x, y);
        const int D_p = hi - lo;
        if (D_p <= 0) return;

        const int i = y * w + x;
        const int x0 = x - Larm[i];
        const int x1 = x + Rarm[i];
        const int arm_span = x1 - x0 + 1;

        std::vector<NeighborInfo> heap_neighbors;
        NeighborInfo stack_neighbors[64];
        NeighborInfo* neighbors = stack_neighbors;
        if (arm_span > 64) {
            heap_neighbors.resize(static_cast<std::size_t>(arm_span));
            neighbors = heap_neighbors.data();
        }

        int num_neighbors = 0;
        for (int xx = x0; xx <= x1; ++xx) {
            const int n_dmin = packed_cost.dmin(xx, y);
            const int n_dmax = packed_cost.dmax(xx, y);
            if (n_dmax > n_dmin) {
                neighbors[num_neighbors++] = {
                    packed_cost.slice(xx, y), n_dmin, n_dmax
                };
            }
        }

        const uint32_t pixel_offset = packed_cost.offset(x, y);
        uint16_t* dst =
            row.data.data() +
            static_cast<std::size_t>(pixel_offset - row_begin(y));

        for (int di = 0; di < D_p; ++di) {
            const int d = lo + di;
            int acc = 0;
            int count = 0;
            for (int k = 0; k < num_neighbors; ++k) {
                if (d >= neighbors[k].dmin && d < neighbors[k].dmax) {
                    const uint16_t c =
                        neighbors[k].slice[d - neighbors[k].dmin];
                    if (c != kInvalidCost) {
                        acc += c;
                        ++count;
                    }
                }
            }
            dst[di] = (count > 0)
                ? static_cast<uint16_t>(acc / count)
                : kInvalidCost;
        }
    };

    auto vertical_pixel = [&](int x, int y) {
        const int lo = packed_cost.dmin(x, y);
        const int hi = packed_cost.dmax(x, y);
        const int D_p = hi - lo;
        if (D_p <= 0) return;

        const int i = y * w + x;
        const int y0 = y - Uarm[i];
        const int y1 = y + Darm[i];
        const int arm_span = y1 - y0 + 1;

        std::vector<NeighborInfo> heap_neighbors;
        NeighborInfo stack_neighbors[64];
        NeighborInfo* neighbors = stack_neighbors;
        if (arm_span > 64) {
            heap_neighbors.resize(static_cast<std::size_t>(arm_span));
            neighbors = heap_neighbors.data();
        }

        int num_neighbors = 0;
        for (int yy = y0; yy <= y1; ++yy) {
            const auto& row =
                ring[static_cast<std::size_t>(yy % ring_rows)];

            const int n_dmin = packed_cost.dmin(x, yy);
            const int n_dmax = packed_cost.dmax(x, yy);
            if (n_dmax > n_dmin) {
                const uint32_t pixel_offset = packed_cost.offset(x, yy);
                const uint16_t* slice =
                    row.data.data() +
                    static_cast<std::size_t>(
                        pixel_offset - row_begin(yy));
                neighbors[num_neighbors++] = {
                    slice, n_dmin, n_dmax
                };
            }
        }

        uint16_t* dst = packed_cost.slice(x, y);
        for (int di = 0; di < D_p; ++di) {
            const int d = lo + di;
            int acc = 0;
            int count = 0;
            for (int k = 0; k < num_neighbors; ++k) {
                if (d >= neighbors[k].dmin && d < neighbors[k].dmax) {
                    const uint16_t c =
                        neighbors[k].slice[d - neighbors[k].dmin];
                    if (c != kInvalidCost) {
                        acc += c;
                        ++count;
                    }
                }
            }
            dst[di] = (count > 0)
                ? static_cast<uint16_t>(acc / count)
                : kInvalidCost;
        }
    };

    const int passes = std::max(1, iterations);
    for (int pass = 0; pass < passes; ++pass) {
#if defined(_OPENMP)
#pragma omp parallel
        {
            for (int block_start = 0;
                 block_start < h;
                 block_start += batch_rows) {
                const int block_end =
                    std::min(h, block_start + batch_rows);

                // Horizontal production for a whole row block.  Flatten
                // (y,x) explicitly so MSVC /openmp, GCC and Clang all use
                // the same OpenMP 2.0-compatible pixel-space schedule.
                const int horiz_rows = block_end - block_start;
                const int horiz_work = horiz_rows * w;
#pragma omp for schedule(static)
                for (int q = 0; q < horiz_work; ++q) {
                    const int ry = q / w;
                    const int x = q - ry * w;
                    const int y = block_start + ry;
                    horizontal_pixel(x, y);
                }

                // Rows whose complete vertical dependency window is now
                // available can be consumed as one block.
                const int vert_begin =
                    std::max(0, block_start - max_down);
                const int vert_end =
                    std::max(0, std::min(h, block_end - max_down));
                const int vert_rows = vert_end - vert_begin;
                const int vert_work = vert_rows * w;

#pragma omp for schedule(static)
                for (int q = 0; q < vert_work; ++q) {
                    const int ry = q / w;
                    const int x = q - ry * w;
                    const int y = vert_begin + ry;
                    vertical_pixel(x, y);
                }
            }

            // Flush the final max_down rows after all horizontal rows exist.
            const int tail_begin = std::max(0, h - max_down);
            const int tail_rows = h - tail_begin;
            const int tail_work = tail_rows * w;
#pragma omp for schedule(static)
            for (int q = 0; q < tail_work; ++q) {
                const int ry = q / w;
                const int x = q - ry * w;
                const int y = tail_begin + ry;
                vertical_pixel(x, y);
            }
        }
#else
        for (int block_start = 0;
             block_start < h;
             block_start += batch_rows) {
            const int block_end =
                std::min(h, block_start + batch_rows);

            for (int y = block_start; y < block_end; ++y) {
                for (int x = 0; x < w; ++x) {
                    horizontal_pixel(x, y);
                }
            }

            const int vert_begin =
                std::max(0, block_start - max_down);
            const int vert_end =
                std::max(0, std::min(h, block_end - max_down));
            for (int y = vert_begin; y < vert_end; ++y) {
                for (int x = 0; x < w; ++x) {
                    vertical_pixel(x, y);
                }
            }
        }

        const int tail_begin = std::max(0, h - max_down);
        for (int y = tail_begin; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                vertical_pixel(x, y);
            }
        }
#endif
    }

    return workspace_bytes;
}

std::size_t CostAggregator::aggregate_packed(const PipelineConfig& cfg,
                                             const Image8& left_gray,
                                             PackedCostVolume16& packed_cost) const {
    if (!cfg.aggregation.enable || packed_cost.empty()) return 0;

    std::vector<int16_t> L, R, U, D;
    build_cross_arms(left_gray, cfg.aggregation, L, R, U, D);
    return aggregate_hv_packed_streaming(
        packed_cost, L, R, U, D, std::max(1, cfg.aggregation.iterations));
}

} // namespace apg
