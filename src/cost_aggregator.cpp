#include "apg_sgm/cost_aggregator.hpp"

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
    CostVolume tmp(w, h, buf.cost.d0(), buf.cost.d0() + D, 0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const int x0 = x - Larm[i];
            const int x1 = x + Rarm[i];
            const int denom = (x1 - x0 + 1);
            uint16_t* dst = tmp.slice(x, y);
            for (int di = 0; di < D; ++di) {
                int acc = 0;
                for (int xx = x0; xx <= x1; ++xx) acc += buf.cost.slice(xx, y)[di];
                dst[di] = static_cast<uint16_t>(acc / denom);
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
            const int denom = (y1 - y0 + 1);
            uint16_t* dst = buf.cost.slice(x, y);
            for (int di = 0; di < D; ++di) {
                int acc = 0;
                for (int yy = y0; yy <= y1; ++yy) acc += tmp.slice(x, yy)[di];
                dst[di] = static_cast<uint16_t>(acc / denom);
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

} // namespace apg
