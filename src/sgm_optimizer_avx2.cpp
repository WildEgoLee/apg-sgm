#include "sgm_optimizer_avx2.hpp"
#include "sgm_common.hpp"

#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace apg {
namespace detail {

namespace {

template <typename TAcc, AccumulateMode Mode>
inline void process_pixel_packed_avx2(int x, int y, bool has_prev, int px, int py,
                                      int P1, const SgmParams& sgm_cfg,
                                      const Image8& gray, const PackedCostVolume16& base,
                                      PackedCostVolume<TAcc>& acc,
                                      PathState& prev, PathState& cur) {
    const int dmin_c = base.dmin(x, y);
    const int dmax_c = base.dmax(x, y);
    const int D_c = dmax_c - dmin_c;
    cur.dmin = dmin_c;
    cur.dmax = dmax_c;
    cur.vals.resize(D_c);
    cur.min_val = kPathInf;

    if (D_c <= 0) {
        std::swap(prev, cur);
        return;
    }

    const uint16_t* c = base.slice(x, y);
    TAcc* a = acc.slice(x, y);
    const TAcc inv_acc = invalid_cost<TAcc>();

    if (!has_prev || prev.vals.empty()) {
        for (int di = 0; di < D_c; ++di) {
            if (c[di] == kInvalidCost) {
                cur.vals[di] = kPathInf;
                a[di] = inv_acc;
            } else {
                cur.vals[di] = static_cast<int>(c[di]);
                if (cur.vals[di] < cur.min_val) {
                    cur.min_val = cur.vals[di];
                }
                if constexpr (Mode == AccumulateMode::Overwrite) {
                    a[di] = static_cast<TAcc>(cur.vals[di]);
                } else {
                    if (a[di] != inv_acc) {
                        a[di] += static_cast<TAcc>(cur.vals[di]);
                    }
                }
            }
        }
        std::swap(prev, cur);
        return;
    }

    const int dI = static_cast<int>(gray.at(x, y)) - static_cast<int>(gray.at(px, py));
    const int P2 = adaptive_p2(sgm_cfg, dI);

    const int dmin_p = prev.dmin;
    const int dmax_p = prev.dmax;
    const int min_prev = prev.min_val;
    const auto& prev_vals = prev.vals;

    auto process_scalar_di = [&](int di) {
        const bool valid_cur = (c[di] != kInvalidCost);
        if (!valid_cur) {
            cur.vals[di] = kPathInf;
            a[di] = inv_acc;
            return;
        }

        const int d = dmin_c + di;
        int best = kPathInf;
        if (d >= dmin_p && d < dmax_p) {
            const int v = prev_vals[d - dmin_p];
            if (v < best) best = v;
        }
        if (d - 1 >= dmin_p && d - 1 < dmax_p) {
            const int v = prev_vals[d - 1 - dmin_p];
            if (v < kPathInf && v + P1 < best) best = v + P1;
        }
        if (d + 1 >= dmin_p && d + 1 < dmax_p) {
            const int v = prev_vals[d + 1 - dmin_p];
            if (v < kPathInf && v + P1 < best) best = v + P1;
        }
        if (min_prev < kPathInf) {
            if (min_prev + P2 < best) best = min_prev + P2;
        }

        if (best >= kPathInf) {
            cur.vals[di] = static_cast<int>(c[di]);
        } else {
            cur.vals[di] = static_cast<int>(c[di]) + best - min_prev;
        }

        if (cur.vals[di] < cur.min_val) {
            cur.min_val = cur.vals[di];
        }

        if constexpr (Mode == AccumulateMode::Overwrite) {
            a[di] = static_cast<TAcc>(cur.vals[di]);
        } else {
            if (a[di] != inv_acc) {
                a[di] += static_cast<TAcc>(cur.vals[di]);
            }
        }
    };

    const int d_int_start = std::max(dmin_c, dmin_p + 1);
    const int d_int_end = std::min(dmax_c, dmax_p - 1);

    if (d_int_start >= d_int_end) {
        for (int di = 0; di < D_c; ++di) {
            process_scalar_di(di);
        }
        std::swap(prev, cur);
        return;
    }

    const int di_int_start = d_int_start - dmin_c;
    const int di_int_end = d_int_end - dmin_c;
    const int int_len = di_int_end - di_int_start;
    const int num_vec8 = (int_len / 8) * 8;
    const int di_vec_end = di_int_start + num_vec8;

    // 1. Scalar prefix
    for (int di = 0; di < di_int_start; ++di) {
        process_scalar_di(di);
    }

    // 2. AVX2 interior
    if (num_vec8 > 0) {
        const int init_best = (min_prev < kPathInf) ? std::min(kPathInf, min_prev + P2) : kPathInf;
        const __m256i v_init_best = _mm256_set1_epi32(init_best);
        const __m256i v_p1 = _mm256_set1_epi32(P1);
        const __m256i v_min_prev = _mm256_set1_epi32(min_prev);
        const __m256i v_cinv16_32 = _mm256_set1_epi32(kInvalidCost);
        const __m256i v_path_inf = _mm256_set1_epi32(kPathInf);
        const __m256i v_ainv32 = _mm256_set1_epi32(kInvalidCost32);
        const bool min_prev_valid = (min_prev < kPathInf);

        __m256i v_min_cur = _mm256_set1_epi32(kPathInf);

        for (int di = di_int_start; di < di_vec_end; di += 8) {
            __m128i c16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + di));
            __m256i vc = _mm256_cvtepu16_epi32(c16);
            __m256i v_is_inv = _mm256_cmpeq_epi32(vc, v_cinv16_32);

            const int j0 = di + (dmin_c - dmin_p);
            const int* p = prev_vals.data() + j0;
            __m256i vp0  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            __m256i vpm1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p - 1));
            __m256i vpp1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 1));

            __m256i v_best = _mm256_min_epi32(v_init_best, vp0);
            v_best = _mm256_min_epi32(v_best, _mm256_add_epi32(vpm1, v_p1));
            v_best = _mm256_min_epi32(v_best, _mm256_add_epi32(vpp1, v_p1));

            __m256i v_cur_raw;
            if (min_prev_valid) {
                __m256i v_diff = _mm256_sub_epi32(v_best, v_min_prev);
                v_cur_raw = _mm256_add_epi32(vc, v_diff);
            } else {
                v_cur_raw = vc;
            }

            __m256i v_cur = _mm256_blendv_epi8(v_cur_raw, v_path_inf, v_is_inv);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(cur.vals.data() + di), v_cur);

            v_min_cur = _mm256_min_epi32(v_min_cur, v_cur);

            if constexpr (std::is_same_v<TAcc, uint16_t>) {
                if constexpr (Mode == AccumulateMode::Overwrite) {
                    __m256i v_new_a32 = _mm256_blendv_epi8(v_cur, v_cinv16_32, v_is_inv);
                    __m128i lo = _mm256_castsi256_si128(v_new_a32);
                    __m128i hi = _mm256_extracti128_si256(v_new_a32, 1);
                    __m128i v_new_a16 = _mm_packus_epi32(lo, hi);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(a + di), v_new_a16);
                } else {
                    __m128i va16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + di));
                    __m256i va = _mm256_cvtepu16_epi32(va16);
                    __m256i va_is_inv = _mm256_cmpeq_epi32(va, v_cinv16_32);
                    __m256i v_any_inv = _mm256_or_si256(v_is_inv, va_is_inv);
                    __m256i va_added = _mm256_add_epi32(va, v_cur);
                    __m256i v_new_a32 = _mm256_blendv_epi8(va_added, v_cinv16_32, v_any_inv);
                    __m128i lo = _mm256_castsi256_si128(v_new_a32);
                    __m128i hi = _mm256_extracti128_si256(v_new_a32, 1);
                    __m128i v_new_a16 = _mm_packus_epi32(lo, hi);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(a + di), v_new_a16);
                }
            } else {
                if constexpr (Mode == AccumulateMode::Overwrite) {
                    __m256i v_new_a = _mm256_blendv_epi8(v_cur, v_ainv32, v_is_inv);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + di), v_new_a);
                } else {
                    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + di));
                    __m256i va_is_inv = _mm256_cmpeq_epi32(va, v_ainv32);
                    __m256i v_any_inv = _mm256_or_si256(v_is_inv, va_is_inv);
                    __m256i va_added = _mm256_add_epi32(va, v_cur);
                    __m256i v_new_a = _mm256_blendv_epi8(va_added, v_ainv32, v_any_inv);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + di), v_new_a);
                }
            }
        }

        __m128i v_low = _mm256_castsi256_si128(v_min_cur);
        __m128i v_high = _mm256_extracti128_si256(v_min_cur, 1);
        __m128i v_m = _mm_min_epi32(v_low, v_high);
        v_m = _mm_min_epi32(v_m, _mm_shuffle_epi32(v_m, _MM_SHUFFLE(1, 0, 3, 2)));
        v_m = _mm_min_epi32(v_m, _mm_shuffle_epi32(v_m, _MM_SHUFFLE(2, 3, 0, 1)));
        int min_from_vec = _mm_cvtsi128_si32(v_m);
        if (min_from_vec < cur.min_val) {
            cur.min_val = min_from_vec;
        }
    }

    // 3. Scalar suffix & interior tail
    for (int di = di_vec_end; di < D_c; ++di) {
        process_scalar_di(di);
    }

    std::swap(prev, cur);
}

template <typename TAcc, AccumulateMode Mode>
void aggregate_path_packed_avx2_impl(const PipelineConfig& cfg, const Image8& gray,
                                     const PackedCostVolume16& base,
                                     PackedCostVolume<TAcc>& acc, int dx, int dy) {
    const int w = base.width();
    const int h = base.height();
    const int P1 = cfg.sgm.P1;

    if (dx == 1 && dy == 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            PathState prev, cur;
            for (int x = 0; x < w; ++x) {
                process_pixel_packed_avx2<TAcc, Mode>(x, y, x > 0, x - 1, y, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == -1 && dy == 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            PathState prev, cur;
            for (int x = w - 1; x >= 0; --x) {
                process_pixel_packed_avx2<TAcc, Mode>(x, y, x + 1 < w, x + 1, y, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == 1) {
        constexpr int kVerticalTileX = 16;
        const int num_blocks = (w + kVerticalTileX - 1) / kVerticalTileX;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int bi = 0; bi < num_blocks; ++bi) {
            const int xb = bi * kVerticalTileX;
            const int xe = std::min(xb + kVerticalTileX, w);
            const int nb = xe - xb;

            std::array<PathState, kVerticalTileX> prev;
            std::array<PathState, kVerticalTileX> cur;

            for (int y = 0; y < h; ++y) {
                const bool has_prev = (y > 0);
                const int py = y - 1;
                for (int i = 0; i < nb; ++i) {
                    const int x = xb + i;
                    process_pixel_packed_avx2<TAcc, Mode>(
                        x, y, has_prev, x, py, P1, cfg.sgm, gray, base, acc, prev[i], cur[i]);
                }
            }
        }
    } else if (dx == 0 && dy == -1) {
        constexpr int kVerticalTileX = 16;
        const int num_blocks = (w + kVerticalTileX - 1) / kVerticalTileX;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int bi = 0; bi < num_blocks; ++bi) {
            const int xb = bi * kVerticalTileX;
            const int xe = std::min(xb + kVerticalTileX, w);
            const int nb = xe - xb;

            std::array<PathState, kVerticalTileX> prev;
            std::array<PathState, kVerticalTileX> cur;

            for (int y = h - 1; y >= 0; --y) {
                const bool has_prev = (y + 1 < h);
                const int py = y + 1;
                for (int i = 0; i < nb; ++i) {
                    const int x = xb + i;
                    process_pixel_packed_avx2<TAcc, Mode>(
                        x, y, has_prev, x, py, P1, cfg.sgm, gray, base, acc, prev[i], cur[i]);
                }
            }
        }
    } else {
        const int nrays = w + h - 1;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int r = 0; r < nrays; ++r) {
            PathState prev, cur;
            int x = 0, y = 0;
            if (dx == 1 && dy == 1) {
                if (r < w) {
                    x = r;
                    y = 0;
                } else {
                    x = 0;
                    y = r - w + 1;
                }
            } else if (dx == 1 && dy == -1) {
                if (r < w) {
                    x = r;
                    y = h - 1;
                } else {
                    x = 0;
                    y = h - 1 - (r - w + 1);
                }
            } else if (dx == -1 && dy == 1) {
                if (r < w) {
                    x = r;
                    y = 0;
                } else {
                    x = w - 1;
                    y = r - w + 1;
                }
            } else { // dx == -1 && dy == -1
                if (r < w) {
                    x = r;
                    y = h - 1;
                } else {
                    x = w - 1;
                    y = h - 1 - (r - w + 1);
                }
            }

            bool hp = false;
            int px = x, py = y;
            while (x >= 0 && x < w && y >= 0 && y < h) {
                process_pixel_packed_avx2<TAcc, Mode>(x, y, hp, px, py, P1, cfg.sgm, gray, base, acc, prev, cur);
                hp = true;
                px = x;
                py = y;
                x += dx;
                y += dy;
            }
        }
    }
}

} // namespace

void aggregate_path_packed_avx2(const PipelineConfig& cfg, const Image8& gray,
                                const PackedCostVolume16& base,
                                PackedCostVolume16& acc, int dx, int dy,
                                AccumulateMode mode) {
    if (mode == AccumulateMode::Overwrite) {
        aggregate_path_packed_avx2_impl<uint16_t, AccumulateMode::Overwrite>(cfg, gray, base, acc, dx, dy);
    } else {
        aggregate_path_packed_avx2_impl<uint16_t, AccumulateMode::Add>(cfg, gray, base, acc, dx, dy);
    }
}

void aggregate_path_packed_avx2(const PipelineConfig& cfg, const Image8& gray,
                                const PackedCostVolume16& base,
                                PackedCostVolume32& acc, int dx, int dy,
                                AccumulateMode mode) {
    if (mode == AccumulateMode::Overwrite) {
        aggregate_path_packed_avx2_impl<uint32_t, AccumulateMode::Overwrite>(cfg, gray, base, acc, dx, dy);
    } else {
        aggregate_path_packed_avx2_impl<uint32_t, AccumulateMode::Add>(cfg, gray, base, acc, dx, dy);
    }
}

} // namespace detail
} // namespace apg
