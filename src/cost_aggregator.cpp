#include "apg_sgm/cost_aggregator.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
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

namespace {

// Prefix entry for a single disparity lane.
// Note: Prefix accumulation runs along the entire row/column dimension.
// Capacity boundary: max_dim * cost_max <= max_dim * 65535.
// For Middlebury dimensions (max_dim ~ 3000), total sum <= 1.96e8 << UINT32_MAX (~4.29e9),
// ensuring uint32_t cannot overflow.
struct PrefixEntry {
    uint32_t sum;
    uint32_t cnt;
};

struct PrefixWorkspace {
    std::vector<PrefixEntry> entries;
    void ensure(size_t max_dim, size_t B) {
        const size_t n = (max_dim + 1) * B;
        if (entries.size() < n) {
            entries.resize(n);
        }
    }
};

} // namespace

void CostAggregator::aggregate_hv_packed(PackedCostVolume16& packed_cost,
                                         PackedCostVolume16& tmp,
                                         const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                                         const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm) const {
    const int w = packed_cost.width();
    const int h = packed_cost.height();
    constexpr int B = 16;

    // Horizontal pass: packed_cost -> tmp
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        PrefixWorkspace ws;
        ws.ensure(static_cast<size_t>(w), B);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
        for (int y = 0; y < h; ++y) {
            int row_lo = std::numeric_limits<int>::max();
            int row_hi = std::numeric_limits<int>::min();
            for (int x = 0; x < w; ++x) {
                const int lo = packed_cost.dmin(x, y);
                const int hi = packed_cost.dmax(x, y);
                if (hi > lo) {
                    row_lo = std::min(row_lo, lo);
                    row_hi = std::max(row_hi, hi);
                }
            }
            if (row_lo >= row_hi) continue;

            for (int td = row_lo; td < row_hi; td += B) {
                const int tb = std::min(B, row_hi - td);

                // Base at x = 0: prefix sum/cnt is 0
                for (int b = 0; b < tb; ++b) {
                    ws.entries[b] = {0, 0};
                }

                // 1D prefix along row y
                for (int x = 0; x < w; ++x) {
                    const int lo = packed_cost.dmin(x, y);
                    const int hi = packed_cost.dmax(x, y);
                    const size_t prev_base = static_cast<size_t>(x) * B;
                    const size_t curr_base = static_cast<size_t>(x + 1) * B;
                    PrefixEntry* const p_prev = ws.entries.data() + prev_base;
                    PrefixEntry* const p_curr = ws.entries.data() + curr_base;

                    const int o_lo = std::max(td, lo);
                    const int o_hi = std::min(td + tb, hi);

                    if (o_lo < o_hi) {
                        const int b_start = o_lo - td;
                        const int b_end = o_hi - td;
                        for (int b = 0; b < b_start; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                        const uint16_t* s = packed_cost.slice(x, y) + (o_lo - lo);
                        for (int b = b_start; b < b_end; ++b) {
                            const uint16_t c = *s++;
                            const uint32_t valid = (c != kInvalidCost);
                            p_curr[b].sum = p_prev[b].sum + (valid ? c : 0);
                            p_curr[b].cnt = p_prev[b].cnt + valid;
                        }
                        for (int b = b_end; b < tb; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                    } else {
                        for (int b = 0; b < tb; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                    }
                }

                // Query for each pixel x in row y
                for (int x = 0; x < w; ++x) {
                    const int lo = packed_cost.dmin(x, y);
                    const int hi = packed_cost.dmax(x, y);
                    const int d_start = std::max(td, lo);
                    const int d_end = std::min(td + tb, hi);
                    if (d_start >= d_end) continue;

                    const int i = y * w + x;
                    const int x0 = x - Larm[i];
                    const int x1 = x + Rarm[i];
                    const PrefixEntry* const p_x0 = ws.entries.data() + static_cast<size_t>(x0) * B;
                    const PrefixEntry* const p_x1 = ws.entries.data() + static_cast<size_t>(x1 + 1) * B;

                    uint16_t* dst = tmp.slice(x, y);
                    for (int d = d_start; d < d_end; ++d) {
                        const int b = d - td;
                        const int di = d - lo;
                        const uint32_t acc = p_x1[b].sum - p_x0[b].sum;
                        const uint32_t cnt = p_x1[b].cnt - p_x0[b].cnt;
                        dst[di] = (cnt > 0) ? static_cast<uint16_t>(acc / cnt) : kInvalidCost;
                    }
                }
            }
        }
    }

    // Vertical pass: tmp -> packed_cost
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        PrefixWorkspace ws;
        ws.ensure(static_cast<size_t>(h), B);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
        for (int x = 0; x < w; ++x) {
            int col_lo = std::numeric_limits<int>::max();
            int col_hi = std::numeric_limits<int>::min();
            for (int y = 0; y < h; ++y) {
                const int lo = tmp.dmin(x, y);
                const int hi = tmp.dmax(x, y);
                if (hi > lo) {
                    col_lo = std::min(col_lo, lo);
                    col_hi = std::max(col_hi, hi);
                }
            }
            if (col_lo >= col_hi) continue;

            for (int td = col_lo; td < col_hi; td += B) {
                const int tb = std::min(B, col_hi - td);

                // Base at y = 0: prefix sum/cnt is 0
                for (int b = 0; b < tb; ++b) {
                    ws.entries[b] = {0, 0};
                }

                // 1D prefix along column x
                for (int y = 0; y < h; ++y) {
                    const int lo = tmp.dmin(x, y);
                    const int hi = tmp.dmax(x, y);
                    const size_t prev_base = static_cast<size_t>(y) * B;
                    const size_t curr_base = static_cast<size_t>(y + 1) * B;
                    PrefixEntry* const p_prev = ws.entries.data() + prev_base;
                    PrefixEntry* const p_curr = ws.entries.data() + curr_base;

                    const int o_lo = std::max(td, lo);
                    const int o_hi = std::min(td + tb, hi);

                    if (o_lo < o_hi) {
                        const int b_start = o_lo - td;
                        const int b_end = o_hi - td;
                        for (int b = 0; b < b_start; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                        const uint16_t* s = tmp.slice(x, y) + (o_lo - lo);
                        for (int b = b_start; b < b_end; ++b) {
                            const uint16_t c = *s++;
                            const uint32_t valid = (c != kInvalidCost);
                            p_curr[b].sum = p_prev[b].sum + (valid ? c : 0);
                            p_curr[b].cnt = p_prev[b].cnt + valid;
                        }
                        for (int b = b_end; b < tb; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                    } else {
                        for (int b = 0; b < tb; ++b) {
                            p_curr[b] = p_prev[b];
                        }
                    }
                }

                // Query for each pixel y in column x
                for (int y = 0; y < h; ++y) {
                    const int lo = packed_cost.dmin(x, y);
                    const int hi = packed_cost.dmax(x, y);
                    const int d_start = std::max(td, lo);
                    const int d_end = std::min(td + tb, hi);
                    if (d_start >= d_end) continue;

                    const int i = y * w + x;
                    const int y0 = y - Uarm[i];
                    const int y1 = y + Darm[i];
                    const PrefixEntry* const p_y0 = ws.entries.data() + static_cast<size_t>(y0) * B;
                    const PrefixEntry* const p_y1 = ws.entries.data() + static_cast<size_t>(y1 + 1) * B;

                    uint16_t* dst = packed_cost.slice(x, y);
                    for (int d = d_start; d < d_end; ++d) {
                        const int b = d - td;
                        const int di = d - lo;
                        const uint32_t acc = p_y1[b].sum - p_y0[b].sum;
                        const uint32_t cnt = p_y1[b].cnt - p_y0[b].cnt;
                        dst[di] = (cnt > 0) ? static_cast<uint16_t>(acc / cnt) : kInvalidCost;
                    }
                }
            }
        }
    }
}

void CostAggregator::aggregate_packed(const PipelineConfig& cfg,
                                      const Image8& left_gray,
                                      PackedCostVolume16& packed_cost) const {
    if (!cfg.aggregation.enable || packed_cost.empty()) return;
    std::vector<int16_t> L, R, U, D;
    build_cross_arms(left_gray, cfg.aggregation, L, R, U, D);
    const int iters = std::max(1, cfg.aggregation.iterations);

    PackedCostVolume16 tmp(packed_cost.layout(), kInvalidCost);
    for (int i = 0; i < iters; ++i) {
        aggregate_hv_packed(packed_cost, tmp, L, R, U, D);
    }
}

} // namespace apg
