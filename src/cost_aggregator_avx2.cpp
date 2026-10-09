#include "cost_aggregator_avx2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg {
namespace detail {

namespace {

struct PrefixWorkspaceSoA {
    std::vector<uint32_t> sums;
    std::vector<uint32_t> cnts;

    void ensure(size_t max_dim, size_t B) {
        const size_t n = (max_dim + 1) * B;
        if (sums.size() < n) {
            sums.resize(n);
            cnts.resize(n);
        }
    }
};

inline __m256i exact_cross_div_8(__m256i v_acc, __m256i v_cnt) {
    const __m256i v_zero = _mm256_setzero_si256();
    __m256i v_safe_cnt = _mm256_max_epi32(v_cnt, _mm256_set1_epi32(1));

    __m256 f_acc = _mm256_cvtepi32_ps(v_acc);
    __m256 f_cnt = _mm256_cvtepi32_ps(v_safe_cnt);
    __m256 f_q = _mm256_div_ps(f_acc, f_cnt);
    __m256i v_q = _mm256_cvttps_epi32(f_q);

    // Exact integer correction:
    __m256i v_prod = _mm256_mullo_epi32(v_q, v_safe_cnt);
    __m256i v_rem = _mm256_sub_epi32(v_acc, v_prod);
    __m256i v_under = _mm256_cmpgt_epi32(v_zero, v_rem);
    v_q = _mm256_add_epi32(v_q, v_under);

    v_prod = _mm256_mullo_epi32(v_q, v_safe_cnt);
    v_rem = _mm256_sub_epi32(v_acc, v_prod);
    __m256i v_rem_sub_cnt = _mm256_sub_epi32(v_rem, v_safe_cnt);
    __m256i v_over = _mm256_cmpgt_epi32(v_zero, v_rem_sub_cnt);
    __m256i v_inc = _mm256_andnot_si256(v_over, _mm256_set1_epi32(1));
    v_q = _mm256_add_epi32(v_q, v_inc);

    // Mask with cnt > 0
    __m256i v_valid = _mm256_cmpgt_epi32(v_cnt, v_zero);
    return _mm256_blendv_epi8(_mm256_set1_epi32(kInvalidCost), v_q, v_valid);
}

inline __m256i pack_16_u32_to_u16(__m256i r0, __m256i r1) {
    __m256i p = _mm256_packus_epi32(r0, r1);
    return _mm256_permute4x64_epi64(p, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename TmpVolume>
void aggregate_hv_packed_avx2_impl(PackedCostVolume16& packed_cost,
                                  TmpVolume& tmp,
                                  const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                                  const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm) {
    const int w = packed_cost.width();
    const int h = packed_cost.height();
    constexpr int B = 16;

    // Horizontal pass: packed_cost -> tmp
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        PrefixWorkspaceSoA ws;
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
                if (tb == 16) {
                    _mm256_storeu_si256((__m256i*)ws.sums.data(), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)(ws.sums.data() + 8), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)ws.cnts.data(), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)(ws.cnts.data() + 8), _mm256_setzero_si256());
                } else {
                    for (int b = 0; b < tb; ++b) {
                        ws.sums[b] = 0;
                        ws.cnts[b] = 0;
                    }
                }

                // 1D prefix along row y
                for (int x = 0; x < w; ++x) {
                    const int lo = packed_cost.dmin(x, y);
                    const int hi = packed_cost.dmax(x, y);
                    const size_t prev_base = static_cast<size_t>(x) * B;
                    const size_t curr_base = static_cast<size_t>(x + 1) * B;
                    const uint32_t* const p_prev_sum = ws.sums.data() + prev_base;
                    const uint32_t* const p_prev_cnt = ws.cnts.data() + prev_base;
                    uint32_t* const p_curr_sum = ws.sums.data() + curr_base;
                    uint32_t* const p_curr_cnt = ws.cnts.data() + curr_base;

                    const int o_lo = std::max(td, lo);
                    const int o_hi = std::min(td + tb, hi);

                    if (o_lo < o_hi) {
                        const int b_start = o_lo - td;
                        const int b_end = o_hi - td;

                        if (tb == 16 && b_start == 0 && b_end == 16) {
                            const uint16_t* s = packed_cost.slice(x, y) + (td - lo);
                            __m128i raw0 = _mm_loadu_si128((const __m128i*)s);
                            __m128i raw1 = _mm_loadu_si128((const __m128i*)(s + 8));
                            __m256i cost0 = _mm256_cvtepu16_epi32(raw0);
                            __m256i cost1 = _mm256_cvtepu16_epi32(raw1);

                            const __m256i v_inv = _mm256_set1_epi32(kInvalidCost);
                            const __m256i v_one = _mm256_set1_epi32(1);

                            __m256i mask_inv0 = _mm256_cmpeq_epi32(cost0, v_inv);
                            __m256i mask_inv1 = _mm256_cmpeq_epi32(cost1, v_inv);

                            __m256i valid_cnt0 = _mm256_andnot_si256(mask_inv0, v_one);
                            __m256i valid_cnt1 = _mm256_andnot_si256(mask_inv1, v_one);

                            __m256i valid_cost0 = _mm256_andnot_si256(mask_inv0, cost0);
                            __m256i valid_cost1 = _mm256_andnot_si256(mask_inv1, cost1);

                            __m256i prev_sum0 = _mm256_loadu_si256((const __m256i*)p_prev_sum);
                            __m256i prev_sum1 = _mm256_loadu_si256((const __m256i*)(p_prev_sum + 8));
                            __m256i prev_cnt0 = _mm256_loadu_si256((const __m256i*)p_prev_cnt);
                            __m256i prev_cnt1 = _mm256_loadu_si256((const __m256i*)(p_prev_cnt + 8));

                            _mm256_storeu_si256((__m256i*)p_curr_sum, _mm256_add_epi32(prev_sum0, valid_cost0));
                            _mm256_storeu_si256((__m256i*)(p_curr_sum + 8), _mm256_add_epi32(prev_sum1, valid_cost1));
                            _mm256_storeu_si256((__m256i*)p_curr_cnt, _mm256_add_epi32(prev_cnt0, valid_cnt0));
                            _mm256_storeu_si256((__m256i*)(p_curr_cnt + 8), _mm256_add_epi32(prev_cnt1, valid_cnt1));
                        } else {
                            for (int b = 0; b < b_start; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
                            const uint16_t* s = packed_cost.slice(x, y) + (o_lo - lo);
                            for (int b = b_start; b < b_end; ++b) {
                                const uint16_t c = *s++;
                                const uint32_t valid = (c != kInvalidCost);
                                p_curr_sum[b] = p_prev_sum[b] + (valid ? c : 0);
                                p_curr_cnt[b] = p_prev_cnt[b] + valid;
                            }
                            for (int b = b_end; b < tb; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
                        }
                    } else {
                        if (tb == 16) {
                            _mm256_storeu_si256((__m256i*)p_curr_sum, _mm256_loadu_si256((const __m256i*)p_prev_sum));
                            _mm256_storeu_si256((__m256i*)(p_curr_sum + 8), _mm256_loadu_si256((const __m256i*)(p_prev_sum + 8)));
                            _mm256_storeu_si256((__m256i*)p_curr_cnt, _mm256_loadu_si256((const __m256i*)p_prev_cnt));
                            _mm256_storeu_si256((__m256i*)(p_curr_cnt + 8), _mm256_loadu_si256((const __m256i*)(p_prev_cnt + 8)));
                        } else {
                            for (int b = 0; b < tb; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
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
                    const uint32_t* const p_x0_sum = ws.sums.data() + static_cast<size_t>(x0) * B;
                    const uint32_t* const p_x0_cnt = ws.cnts.data() + static_cast<size_t>(x0) * B;
                    const uint32_t* const p_x1_sum = ws.sums.data() + static_cast<size_t>(x1 + 1) * B;
                    const uint32_t* const p_x1_cnt = ws.cnts.data() + static_cast<size_t>(x1 + 1) * B;

                    uint16_t* dst = tmp.slice(x, y);
                    if (tb == 16 && d_start == td && d_end == td + 16) {
                        __m256i acc0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)p_x1_sum),
                                                        _mm256_loadu_si256((const __m256i*)p_x0_sum));
                        __m256i acc1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(p_x1_sum + 8)),
                                                        _mm256_loadu_si256((const __m256i*)(p_x0_sum + 8)));

                        __m256i cnt0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)p_x1_cnt),
                                                        _mm256_loadu_si256((const __m256i*)p_x0_cnt));
                        __m256i cnt1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(p_x1_cnt + 8)),
                                                        _mm256_loadu_si256((const __m256i*)(p_x0_cnt + 8)));

                        __m256i q0 = exact_cross_div_8(acc0, cnt0);
                        __m256i q1 = exact_cross_div_8(acc1, cnt1);

                        __m256i packed_res = pack_16_u32_to_u16(q0, q1);
                        const int di = td - lo;
                        _mm256_storeu_si256((__m256i*)(dst + di), packed_res);
                    } else {
                        for (int d = d_start; d < d_end; ++d) {
                            const int b = d - td;
                            const int di = d - lo;
                            const uint32_t acc = p_x1_sum[b] - p_x0_sum[b];
                            const uint32_t cnt = p_x1_cnt[b] - p_x0_cnt[b];
                            dst[di] = (cnt > 0) ? static_cast<uint16_t>(acc / cnt) : kInvalidCost;
                        }
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
        PrefixWorkspaceSoA ws;
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
                if (tb == 16) {
                    _mm256_storeu_si256((__m256i*)ws.sums.data(), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)(ws.sums.data() + 8), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)ws.cnts.data(), _mm256_setzero_si256());
                    _mm256_storeu_si256((__m256i*)(ws.cnts.data() + 8), _mm256_setzero_si256());
                } else {
                    for (int b = 0; b < tb; ++b) {
                        ws.sums[b] = 0;
                        ws.cnts[b] = 0;
                    }
                }

                // 1D prefix along column x
                for (int y = 0; y < h; ++y) {
                    const int lo = tmp.dmin(x, y);
                    const int hi = tmp.dmax(x, y);
                    const size_t prev_base = static_cast<size_t>(y) * B;
                    const size_t curr_base = static_cast<size_t>(y + 1) * B;
                    const uint32_t* const p_prev_sum = ws.sums.data() + prev_base;
                    const uint32_t* const p_prev_cnt = ws.cnts.data() + prev_base;
                    uint32_t* const p_curr_sum = ws.sums.data() + curr_base;
                    uint32_t* const p_curr_cnt = ws.cnts.data() + curr_base;

                    const int o_lo = std::max(td, lo);
                    const int o_hi = std::min(td + tb, hi);

                    if (o_lo < o_hi) {
                        const int b_start = o_lo - td;
                        const int b_end = o_hi - td;

                        if (tb == 16 && b_start == 0 && b_end == 16) {
                            const uint16_t* s = tmp.slice(x, y) + (td - lo);
                            __m128i raw0 = _mm_loadu_si128((const __m128i*)s);
                            __m128i raw1 = _mm_loadu_si128((const __m128i*)(s + 8));
                            __m256i cost0 = _mm256_cvtepu16_epi32(raw0);
                            __m256i cost1 = _mm256_cvtepu16_epi32(raw1);

                            const __m256i v_inv = _mm256_set1_epi32(kInvalidCost);
                            const __m256i v_one = _mm256_set1_epi32(1);

                            __m256i mask_inv0 = _mm256_cmpeq_epi32(cost0, v_inv);
                            __m256i mask_inv1 = _mm256_cmpeq_epi32(cost1, v_inv);

                            __m256i valid_cnt0 = _mm256_andnot_si256(mask_inv0, v_one);
                            __m256i valid_cnt1 = _mm256_andnot_si256(mask_inv1, v_one);

                            __m256i valid_cost0 = _mm256_andnot_si256(mask_inv0, cost0);
                            __m256i valid_cost1 = _mm256_andnot_si256(mask_inv1, cost1);

                            __m256i prev_sum0 = _mm256_loadu_si256((const __m256i*)p_prev_sum);
                            __m256i prev_sum1 = _mm256_loadu_si256((const __m256i*)(p_prev_sum + 8));
                            __m256i prev_cnt0 = _mm256_loadu_si256((const __m256i*)p_prev_cnt);
                            __m256i prev_cnt1 = _mm256_loadu_si256((const __m256i*)(p_prev_cnt + 8));

                            _mm256_storeu_si256((__m256i*)p_curr_sum, _mm256_add_epi32(prev_sum0, valid_cost0));
                            _mm256_storeu_si256((__m256i*)(p_curr_sum + 8), _mm256_add_epi32(prev_sum1, valid_cost1));
                            _mm256_storeu_si256((__m256i*)p_curr_cnt, _mm256_add_epi32(prev_cnt0, valid_cnt0));
                            _mm256_storeu_si256((__m256i*)(p_curr_cnt + 8), _mm256_add_epi32(prev_cnt1, valid_cnt1));
                        } else {
                            for (int b = 0; b < b_start; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
                            const uint16_t* s = tmp.slice(x, y) + (o_lo - lo);
                            for (int b = b_start; b < b_end; ++b) {
                                const uint16_t c = *s++;
                                const uint32_t valid = (c != kInvalidCost);
                                p_curr_sum[b] = p_prev_sum[b] + (valid ? c : 0);
                                p_curr_cnt[b] = p_prev_cnt[b] + valid;
                            }
                            for (int b = b_end; b < tb; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
                        }
                    } else {
                        if (tb == 16) {
                            _mm256_storeu_si256((__m256i*)p_curr_sum, _mm256_loadu_si256((const __m256i*)p_prev_sum));
                            _mm256_storeu_si256((__m256i*)(p_curr_sum + 8), _mm256_loadu_si256((const __m256i*)(p_prev_sum + 8)));
                            _mm256_storeu_si256((__m256i*)p_curr_cnt, _mm256_loadu_si256((const __m256i*)p_prev_cnt));
                            _mm256_storeu_si256((__m256i*)(p_curr_cnt + 8), _mm256_loadu_si256((const __m256i*)(p_prev_cnt + 8)));
                        } else {
                            for (int b = 0; b < tb; ++b) {
                                p_curr_sum[b] = p_prev_sum[b];
                                p_curr_cnt[b] = p_prev_cnt[b];
                            }
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
                    const uint32_t* const p_y0_sum = ws.sums.data() + static_cast<size_t>(y0) * B;
                    const uint32_t* const p_y0_cnt = ws.cnts.data() + static_cast<size_t>(y0) * B;
                    const uint32_t* const p_y1_sum = ws.sums.data() + static_cast<size_t>(y1 + 1) * B;
                    const uint32_t* const p_y1_cnt = ws.cnts.data() + static_cast<size_t>(y1 + 1) * B;

                    uint16_t* dst = packed_cost.slice(x, y);
                    if (tb == 16 && d_start == td && d_end == td + 16) {
                        __m256i acc0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)p_y1_sum),
                                                        _mm256_loadu_si256((const __m256i*)p_y0_sum));
                        __m256i acc1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(p_y1_sum + 8)),
                                                        _mm256_loadu_si256((const __m256i*)(p_y0_sum + 8)));

                        __m256i cnt0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)p_y1_cnt),
                                                        _mm256_loadu_si256((const __m256i*)p_y0_cnt));
                        __m256i cnt1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(p_y1_cnt + 8)),
                                                        _mm256_loadu_si256((const __m256i*)(p_y0_cnt + 8)));

                        __m256i q0 = exact_cross_div_8(acc0, cnt0);
                        __m256i q1 = exact_cross_div_8(acc1, cnt1);

                        __m256i packed_res = pack_16_u32_to_u16(q0, q1);
                        const int di = td - lo;
                        _mm256_storeu_si256((__m256i*)(dst + di), packed_res);
                    } else {
                        for (int d = d_start; d < d_end; ++d) {
                            const int b = d - td;
                            const int di = d - lo;
                            const uint32_t acc = p_y1_sum[b] - p_y0_sum[b];
                            const uint32_t cnt = p_y1_cnt[b] - p_y0_cnt[b];
                            dst[di] = (cnt > 0) ? static_cast<uint16_t>(acc / cnt) : kInvalidCost;
                        }
                    }
                }
            }
        }
    }
}

} // namespace

void aggregate_hv_packed_avx2(PackedCostVolume16& packed_cost,
                              PackedCrossTmp& tmp,
                              const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                              const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm) {
    aggregate_hv_packed_avx2_impl(packed_cost, tmp, Larm, Rarm, Uarm, Darm);
}

void aggregate_hv_packed_avx2(PackedCostVolume16& packed_cost,
                              PackedCostVolume16& tmp,
                              const std::vector<int16_t>& Larm, const std::vector<int16_t>& Rarm,
                              const std::vector<int16_t>& Uarm, const std::vector<int16_t>& Darm) {
    aggregate_hv_packed_avx2_impl(packed_cost, tmp, Larm, Rarm, Uarm, Darm);
}

} // namespace detail
} // namespace apg
