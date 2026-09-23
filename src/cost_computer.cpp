#include "apg_sgm/cost_computer.hpp"

#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace apg {

int CostComputer::popcount32(uint32_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#endif
}

int CostComputer::popcount64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    return popcount32(static_cast<uint32_t>(x)) + popcount32(static_cast<uint32_t>(x >> 32));
#endif
}

uint32_t CostComputer::symmetric_census9x7(const Image8& gray, int x, int y) {
    uint32_t bits = 0;
    int b = 0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -4; dx <= 4; ++dx) {
            if (dy < 0 || (dy == 0 && dx < 0)) {
                const uint8_t a = gray.sample(x + dx, y + dy);
                const uint8_t c = gray.sample(x - dx, y - dy);
                if (a > c) bits |= (1u << b);
                ++b;
            }
        }
    }
    return bits;
}

uint64_t CostComputer::census9x7(const Image8& gray, int x, int y) {
    const uint8_t center = gray.sample(x, y);
    uint64_t bits = 0;
    int b = 0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -4; dx <= 4; ++dx) {
            if (dx == 0 && dy == 0) continue;
            if (gray.sample(x + dx, y + dy) > center) bits |= (1ull << b);
            ++b;
        }
    }
    return bits;
}

void CostComputer::build_gray_and_grad(const Image8& src, Image8& gray, Image8& gx, Image8& gy) const {
    gray = Image8(src.width(), src.height(), 1);
    gx = Image8(src.width(), src.height(), 1);
    gy = Image8(src.width(), src.height(), 1);
    const int w = src.width();
    const int h = src.height();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            gray.at(x, y) = src.gray(x, y);
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int gxv = static_cast<int>(gray.sample(x + 1, y)) - static_cast<int>(gray.sample(x - 1, y));
            const int gyv = static_cast<int>(gray.sample(x, y + 1)) - static_cast<int>(gray.sample(x, y - 1));
            gx.at(x, y) = static_cast<uint8_t>(clampi(std::abs(gxv), 0, 255));
            gy.at(x, y) = static_cast<uint8_t>(clampi(std::abs(gyv), 0, 255));
        }
    }
}

void CostComputer::build_census(const Image8& gray, CensusType type,
                                std::vector<uint32_t>& c32, std::vector<uint64_t>& c64) const {
    const int w = gray.width();
    const int h = gray.height();
    const int n = w * h;
    if (type == CensusType::SymmetricCensus9x7) {
        c32.assign(n, 0);
        c64.clear();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                c32[y * w + x] = symmetric_census9x7(gray, x, y);
            }
        }
    } else {
        c64.assign(n, 0);
        c32.clear();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                c64[y * w + x] = census9x7(gray, x, y);
            }
        }
    }
}

void CostComputer::compute_aux(const Image8& left, const Image8& right, const PipelineConfig& cfg,
                               PipelineBuffers& buf) const {
    buf.left = left;
    buf.right = right;
    build_gray_and_grad(left, buf.left_gray, buf.left_gx, buf.left_gy);
    build_gray_and_grad(right, buf.right_gray, buf.right_gx, buf.right_gy);
    build_census(buf.left_gray, cfg.cost.census, buf.census_left, buf.census_left64);
    build_census(buf.right_gray, cfg.cost.census, buf.census_right, buf.census_right64);
}

void CostComputer::compute_volume(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.left_gray.width();
    const int h = buf.left_gray.height();
    const int d0 = cfg.min_disparity;
    const int d1 = cfg.max_disparity;
    buf.cost.allocate(w, h, d0, d1, static_cast<uint16_t>(cfg.cost.cost_max));

    const bool sym = cfg.cost.census == CensusType::SymmetricCensus9x7;
    const bool use_ad = cfg.cost.use_ad;
    const bool use_grad = cfg.cost.use_grad;
    const float lc = std::max(cfg.cost.lambda_census, 1e-3f);
    const float la = std::max(cfg.cost.lambda_ad, 1e-3f);
    const float lg = std::max(cfg.cost.lambda_grad, 1e-3f);
    const float eta = cfg.cost.eta_ad;
    const float mu = cfg.cost.mu_grad;
    const float scale = static_cast<float>(cfg.cost.cost_max);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = buf.range.dmin.at(x, y);
            const int hi = buf.range.dmax.at(x, y);
            uint16_t* slice = buf.cost.slice(x, y);
            for (int d = d0; d < d1; ++d) {
                const int di = d - d0;
                if (d < lo || d >= hi) {
                    slice[di] = static_cast<uint16_t>(cfg.cost.cost_max);
                    continue;
                }
                const int xr = x - d;
                if (xr < 0 || xr >= w) {
                    slice[di] = static_cast<uint16_t>(cfg.cost.cost_max);
                    continue;
                }
                float ccensus = 0.f;
                if (sym) {
                    ccensus = static_cast<float>(
                        popcount32(buf.census_left[y * w + x] ^ buf.census_right[y * w + xr]));
                } else {
                    ccensus = static_cast<float>(
                        popcount64(buf.census_left64[y * w + x] ^ buf.census_right64[y * w + xr]));
                }
                float c = 1.f - std::exp(-ccensus / lc);
                if (use_ad) {
                    const int ad = std::abs(static_cast<int>(buf.left_gray.at(x, y)) -
                                            static_cast<int>(buf.right_gray.at(xr, y)));
                    c += eta * (1.f - std::exp(-static_cast<float>(ad) / la));
                }
                if (use_grad) {
                    const int g =
                        std::abs(static_cast<int>(buf.left_gx.at(x, y)) -
                                 static_cast<int>(buf.right_gx.at(xr, y))) +
                        std::abs(static_cast<int>(buf.left_gy.at(x, y)) -
                                 static_cast<int>(buf.right_gy.at(xr, y)));
                    c += mu * (1.f - std::exp(-static_cast<float>(g) / lg));
                }
                const int q = static_cast<int>(c * scale / (1.f + eta + mu) + 0.5f);
                slice[di] = static_cast<uint16_t>(clampi(q, 0, cfg.cost.cost_max));
            }
        }
    }
}

} // namespace apg
