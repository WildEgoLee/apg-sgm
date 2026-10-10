#include "cost_computer_avx2.hpp"
#include "discrete_cost_lut.hpp"
#include "apg_sgm/cost_computer.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg::detail {

namespace {

inline __m256i popcount32_avx2(__m256i v) {
    alignas(32) static const uint8_t popcnt_lut16[32] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    };
    const __m256i v_lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(popcnt_lut16));
    const __m256i v_low_mask = _mm256_set1_epi8(0x0F);

    const __m256i v_low = _mm256_and_si256(v, v_low_mask);
    const __m256i v_high = _mm256_and_si256(_mm256_srli_epi16(v, 4), v_low_mask);

    const __m256i cnt_low = _mm256_shuffle_epi8(v_lut, v_low);
    const __m256i cnt_high = _mm256_shuffle_epi8(v_lut, v_high);

    const __m256i cnt_bytes = _mm256_add_epi8(cnt_low, cnt_high);

    const __m256i ones_byte = _mm256_set1_epi16(0x0101);
    const __m256i cnt_words = _mm256_maddubs_epi16(cnt_bytes, ones_byte);

    const __m256i ones_word = _mm256_set1_epi32(0x00010001);
    const __m256i cnt_dwords = _mm256_madd_epi16(cnt_words, ones_word);

    return cnt_dwords;
}

template <bool UseAd, bool UseGrad>
inline void evaluate_scalar_cost_state_t(
    const PipelineConfig& cfg,
    const PipelineBuffers& buf,
    const detail::DiscreteCostLut& lut,
    float scale,
    int x, int y, size_t row_offset,
    int xr, int w,
    uint16_t* slice, int di)
{
    if (xr < 0 || xr >= w) {
        slice[di] = kInvalidCost;
        return;
    }
    const int pop = CostComputer::popcount32(buf.census_left[row_offset + x] ^ buf.census_right[row_offset + xr]);
    float c = lut.census[pop];
    if constexpr (UseAd) {
        const int ad = std::abs(static_cast<int>(buf.left_gray.at(x, y)) -
                                static_cast<int>(buf.right_gray.at(xr, y)));
        c += lut.ad[ad];
    }
    if constexpr (UseGrad) {
        const int g = std::abs(static_cast<int>(buf.left_gx.at(x, y)) -
                               static_cast<int>(buf.right_gx.at(xr, y))) +
                      std::abs(static_cast<int>(buf.left_gy.at(x, y)) -
                               static_cast<int>(buf.right_gy.at(xr, y)));
        c += lut.grad[g];
    }
    const int q = static_cast<int>(c * scale + 0.5f);
    slice[di] = static_cast<uint16_t>(clampi(q, 0, cfg.cost.cost_max));
}

template <bool UseAd, bool UseGrad>
void compute_volume_packed_avx2_impl(
    const PipelineConfig& cfg,
    const PipelineBuffers& buf,
    PackedCostVolume16& packed_cost)
{
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();

    const float eta = cfg.cost.eta_ad;
    const float mu = cfg.cost.mu_grad;
    const float norm = 1.f + (UseAd ? eta : 0.f) + (UseGrad ? mu : 0.f);
    const float scale = static_cast<float>(cfg.cost.cost_max) / std::max(norm, 1e-6f);

    const detail::DiscreteCostLut lut(cfg);

    const __m256 v_scale = _mm256_set1_ps(scale);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256i v_zero = _mm256_setzero_si256();
    const __m256i v_cmax = _mm256_set1_epi32(cfg.cost.cost_max);
    const __m256i v_rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m128i bswap_mask = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int y = 0; y < h; ++y) {
        const size_t row_offset = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const int lo = packed_cost.dmin(x, y);
            const int hi = packed_cost.dmax(x, y);
            const int D_p = (hi > lo) ? (hi - lo) : 0;
            if (D_p <= 0) continue;

            uint16_t* slice = packed_cost.slice(x, y);

            // Vector-safe bounds calculation:
            // For 8-lane block at di:
            // xr_hi = x - lo - di, xr_lo = x - lo - di - 7
            // Safe range requires: xr_lo >= 0 and xr_hi < w
            // <=> di <= x - lo - 7 and di >= x - lo - w + 1
            const int di_vstart = std::max(0, x - lo - w + 1);
            const int di_vend = std::min(D_p, x - lo + 1);

            int di = 0;
            // Prefix before vector-safe region
            while (di < di_vstart && di < D_p) {
                const int d = lo + di;
                const int xr = x - d;
                evaluate_scalar_cost_state_t<UseAd, UseGrad>(cfg, buf, lut, scale, x, y, row_offset, xr, w, slice, di);
                ++di;
            }

            const int v_end = di_vstart + (std::max(0, di_vend - di_vstart) / 8) * 8;

            if (di < v_end) {
                const uint32_t c_l = buf.census_left[row_offset + x];
                const __m256i vc_l = _mm256_set1_epi32(c_l);

                const int left_g = UseAd ? buf.left_gray.at(x, y) : 0;
                const __m256i vg_l = UseAd ? _mm256_set1_epi32(left_g) : v_zero;

                const int left_gx = UseGrad ? buf.left_gx.at(x, y) : 0;
                const int left_gy = UseGrad ? buf.left_gy.at(x, y) : 0;
                const __m256i vgx_l = UseGrad ? _mm256_set1_epi32(left_gx) : v_zero;
                const __m256i vgy_l = UseGrad ? _mm256_set1_epi32(left_gy) : v_zero;

                for (; di < v_end; di += 8) {
                    const int d0 = lo + di;
                    const int xr0 = x - d0;

                    // 1. Census
                    const uint32_t* p_c_r = &buf.census_right[row_offset + xr0 - 7];
                    __m256i vc_r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_c_r));
                    __m256i vc_r = _mm256_permutevar8x32_epi32(vc_r_raw, v_rev8);
                    __m256i v_diff = _mm256_xor_si256(vc_l, vc_r);
                    __m256i v_pop = popcount32_avx2(v_diff);

                    __m256 vc = _mm256_i32gather_ps(lut.census, v_pop, 4);

                    // 2. Gray AD
                    if constexpr (UseAd) {
                        const uint8_t* p_g_r = buf.right_gray.data() + row_offset + xr0 - 7;
                        __m128i raw_g = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p_g_r));
                        __m256i vg_r = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(raw_g, bswap_mask));
                        __m256i vad = _mm256_abs_epi32(_mm256_sub_epi32(vg_l, vg_r));
                        __m256 v_ad_cost = _mm256_i32gather_ps(lut.ad, vad, 4);
                        vc = _mm256_add_ps(vc, v_ad_cost);
                    }

                    // 3. Gradient
                    if constexpr (UseGrad) {
                        const uint8_t* p_gx_r = buf.right_gx.data() + row_offset + xr0 - 7;
                        __m128i raw_gx = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p_gx_r));
                        __m256i vgx_r = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(raw_gx, bswap_mask));
                        __m256i vdiff_gx = _mm256_abs_epi32(_mm256_sub_epi32(vgx_l, vgx_r));

                        const uint8_t* p_gy_r = buf.right_gy.data() + row_offset + xr0 - 7;
                        __m128i raw_gy = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p_gy_r));
                        __m256i vgy_r = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(raw_gy, bswap_mask));
                        __m256i vdiff_gy = _mm256_abs_epi32(_mm256_sub_epi32(vgy_l, vgy_r));

                        __m256i vg = _mm256_add_epi32(vdiff_gx, vdiff_gy);
                        __m256 v_grad_cost = _mm256_i32gather_ps(lut.grad, vg, 4);
                        vc = _mm256_add_ps(vc, v_grad_cost);
                    }

                    // 4. Quantization & Saturation
                    __m256 vq = _mm256_mul_ps(vc, v_scale);
                    vq = _mm256_add_ps(vq, v_half);
                    __m256i vqi = _mm256_cvttps_epi32(vq);
                    vqi = _mm256_max_epi32(v_zero, _mm256_min_epi32(vqi, v_cmax));

                    __m128i low128 = _mm256_castsi256_si128(vqi);
                    __m128i high128 = _mm256_extracti128_si256(vqi, 1);
                    __m128i v_pack = _mm_packus_epi32(low128, high128);

                    _mm_storeu_si128(reinterpret_cast<__m128i*>(slice + di), v_pack);
                }
            }

            // Scalar tail (remaining valid states and states after di_vend)
            for (; di < D_p; ++di) {
                const int d = lo + di;
                const int xr = x - d;
                evaluate_scalar_cost_state_t<UseAd, UseGrad>(cfg, buf, lut, scale, x, y, row_offset, xr, w, slice, di);
            }
        }
    }
}

} // namespace

void compute_volume_packed_avx2(
    const PipelineConfig& cfg,
    const PipelineBuffers& buf,
    PackedCostVolume16& packed_cost)
{
    if (cfg.cost.use_ad && cfg.cost.use_grad) {
        compute_volume_packed_avx2_impl<true, true>(cfg, buf, packed_cost);
    } else if (cfg.cost.use_ad && !cfg.cost.use_grad) {
        compute_volume_packed_avx2_impl<true, false>(cfg, buf, packed_cost);
    } else if (!cfg.cost.use_ad && cfg.cost.use_grad) {
        compute_volume_packed_avx2_impl<false, true>(cfg, buf, packed_cost);
    } else {
        compute_volume_packed_avx2_impl<false, false>(cfg, buf, packed_cost);
    }
}

void build_symmetric_census9x7_avx2(
    const Image8& gray,
    std::vector<uint32_t>& c32)
{
    const int w = gray.width();
    const int h = gray.height();
    c32.assign(w * h, 0);
    const uint8_t* ptr = gray.data();

    // 31 symmetric pairs in identical order as CostComputer::symmetric_census9x7
    struct PairOffset {
        int dx;
        int dy;
    };
    static const auto pairs = []() {
        std::vector<PairOffset> p;
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -4; dx <= 4; ++dx) {
                if (dy < 0 || (dy == 0 && dx < 0)) {
                    p.push_back({dx, dy});
                }
            }
        }
        return p;
    }();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
        // Top and bottom border rows: y < 3 || y >= h - 3
        if (y < 3 || y >= h - 3) {
            for (int x = 0; x < w; ++x) {
                c32[y * w + x] = CostComputer::symmetric_census9x7(gray, x, y);
            }
            continue;
        }

        // Left border: x in [0, 3]
        for (int x = 0; x < std::min(4, w); ++x) {
            c32[y * w + x] = CostComputer::symmetric_census9x7(gray, x, y);
        }

        const int row_offset = y * w;
        const int x_interior_start = 4;
        const int x_interior_end = w - 5; // inclusive

        if (x_interior_start <= x_interior_end) {
            const int interior_len = x_interior_end - x_interior_start + 1;
            const int vec_end = x_interior_start + (interior_len / 8) * 8;

            // Interior AVX2 vector loop: 8 pixels per iteration
            for (int x = x_interior_start; x < vec_end; x += 8) {
                const uint8_t* center = ptr + row_offset + x;
                __m256i accum = _mm256_setzero_si256();

                for (int b = 0; b < 31; ++b) {
                    const int off_a = pairs[b].dy * w + pairs[b].dx;
                    const int off_c = -pairs[b].dy * w - pairs[b].dx;

                    // Load 8 bytes (64 bits) for 8 consecutive pixels
                    __m128i raw_a = _mm_loadu_si64(center + off_a);
                    __m128i raw_c = _mm_loadu_si64(center + off_c);

                    // Zero-extend uint8 to uint32
                    __m256i val_a = _mm256_cvtepu8_epi32(raw_a);
                    __m256i val_c = _mm256_cvtepu8_epi32(raw_c);

                    // a > c comparison (values in [0, 255], signed 32-bit cmpgt is exact)
                    __m256i cmp = _mm256_cmpgt_epi32(val_a, val_c);

                    // Mask with bit value (1u << b)
                    __m256i bit_val = _mm256_set1_epi32(1u << b);
                    __m256i masked = _mm256_and_si256(cmp, bit_val);

                    accum = _mm256_or_si256(accum, masked);
                }

                // Store 8 x uint32 descriptors
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&c32[row_offset + x]), accum);
            }

            // Scalar interior tail and right border
            for (int x = vec_end; x < w; ++x) {
                c32[row_offset + x] = CostComputer::symmetric_census9x7(gray, x, y);
            }
        } else {
            // Very narrow image where interior is empty
            for (int x = 4; x < w; ++x) {
                c32[row_offset + x] = CostComputer::symmetric_census9x7(gray, x, y);
            }
        }
    }
}

} // namespace apg::detail
