#include "apg_sgm/sgm_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace apg {

namespace {

int adaptive_p2(const SgmParams& p, int dI) {
    dI = std::abs(dI);
    int P2 = p.P2_base;
    if (!p.adaptive_penalties) {
        return std::max(P2, p.P1 + 1);
    }
    if (p.piecewise_p2) {
        if (dI >= p.grad_t2) P2 = p.P2_base / 4;
        else if (dI >= p.grad_t1) P2 = p.P2_base / 2;
        else P2 = p.P2_base;
    } else {
        P2 = static_cast<int>(p.P2_base / (1.f + p.p2_alpha * static_cast<float>(dI)) + 0.5f);
    }
    return std::max(P2, p.P1 + 1);
}

} // namespace

void SgmOptimizer::aggregate_path(const PipelineConfig& cfg, const Image8& gray,
                                  CostVolume& vol, int dx, int dy) const {
    const int w = vol.width();
    const int h = vol.height();
    const int D = vol.D();
    const int P1 = cfg.sgm.P1;

    std::vector<int> prev(D), cur(D);

    auto process = [&](int x, int y, bool has_prev, int px, int py) {
        uint16_t* c = vol.slice(x, y);
        if (!has_prev) {
            for (int d = 0; d < D; ++d) prev[d] = c[d];
            return;
        }
        const int dI = static_cast<int>(gray.at(x, y)) - static_cast<int>(gray.at(px, py));
        const int P2 = adaptive_p2(cfg.sgm, dI);
        int min_prev = prev[0];
        for (int d = 1; d < D; ++d) min_prev = std::min(min_prev, prev[d]);
        for (int d = 0; d < D; ++d) {
            int best = prev[d];
            if (d > 0) best = std::min(best, prev[d - 1] + P1);
            if (d + 1 < D) best = std::min(best, prev[d + 1] + P1);
            best = std::min(best, min_prev + P2);
            int v = static_cast<int>(c[d]) + best - min_prev;
            if (v < 0) v = 0;
            if (v > 65535) v = 65535;
            cur[d] = v;
            c[d] = static_cast<uint16_t>(v);
        }
        prev.swap(cur);
    };

    if (dx == 1 && dy == 0) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) process(x, y, x > 0, x - 1, y);
        }
    } else if (dx == -1 && dy == 0) {
        for (int y = 0; y < h; ++y) {
            for (int x = w - 1; x >= 0; --x) process(x, y, x + 1 < w, x + 1, y);
        }
    } else if (dx == 0 && dy == 1) {
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) process(x, y, y > 0, x, y - 1);
        }
    } else if (dx == 0 && dy == -1) {
        for (int x = 0; x < w; ++x) {
            for (int y = h - 1; y >= 0; --y) process(x, y, y + 1 < h, x, y + 1);
        }
    } else {
        std::vector<uint8_t> seen(static_cast<size_t>(w) * h, 0);
        auto inb = [&](int x, int y) { return x >= 0 && x < w && y >= 0 && y < h; };
        for (int y0 = 0; y0 < h; ++y0) {
            for (int x0 = 0; x0 < w; ++x0) {
                if (seen[y0 * w + x0]) continue;
                int x = x0, y = y0;
                while (inb(x - dx, y - dy)) {
                    x -= dx;
                    y -= dy;
                }
                bool hp = false;
                int px = x, py = y;
                while (inb(x, y)) {
                    seen[y * w + x] = 1;
                    process(x, y, hp, px, py);
                    hp = true;
                    px = x;
                    py = y;
                    x += dx;
                    y += dy;
                }
            }
        }
    }
}

void SgmOptimizer::optimize(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.cost.width();
    const int h = buf.cost.height();
    const int D = buf.cost.D();
    const CostVolume base = buf.cost;
    CostVolume acc(w, h, buf.cost.d0(), buf.cost.d0() + D, 0);

    const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const int npath = (cfg.sgm.paths == PathType::Path8) ? 8 : 4;

    for (int p = 0; p < npath; ++p) {
        CostVolume work = base;
        aggregate_path(cfg, buf.left_gray, work, dirs[p][0], dirs[p][1]);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                uint16_t* a = acc.slice(x, y);
                const uint16_t* s = work.slice(x, y);
                for (int d = 0; d < D; ++d) {
                    int v = static_cast<int>(a[d]) + s[d];
                    if (v > 65535) v = 65535;
                    a[d] = static_cast<uint16_t>(v);
                }
            }
        }
    }
    buf.cost = std::move(acc);
}

void SgmOptimizer::winner_take_all(const PipelineConfig& cfg, const CostVolume& vol,
                                   const SearchRange& range, Image32f& disp_out,
                                   Image32f* best_cost, Image32f* second_cost) const {
    const int w = vol.width();
    const int h = vol.height();
    const int d0 = vol.d0();
    const int D = vol.D();
    disp_out = Image32f(w, h, -1.f);
    if (best_cost) *best_cost = Image32f(w, h, 1e9f);
    if (second_cost) *second_cost = Image32f(w, h, 1e9f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint16_t* s = vol.slice(x, y);
            const int lo = range.dmin.at(x, y);
            const int hi = range.dmax.at(x, y);
            int best_d = lo;
            int best = 65535;
            int second = 65535;
            for (int d = lo; d < hi; ++d) {
                const int di = d - d0;
                if (di < 0 || di >= D) continue;
                const int c = s[di];
                if (c < best) {
                    second = best;
                    best = c;
                    best_d = d;
                } else if (c < second) {
                    second = c;
                }
            }
            float dval = static_cast<float>(best_d);
            if (cfg.post.subpixel) {
                const int di = best_d - d0;
                if (di > 0 && di + 1 < D) {
                    const float cm = static_cast<float>(s[di - 1]);
                    const float c0 = static_cast<float>(s[di]);
                    const float cp = static_cast<float>(s[di + 1]);
                    const float denom = cm - 2.f * c0 + cp;
                    if (std::abs(denom) > 1e-6f) {
                        float delta = 0.5f * (cm - cp) / denom;
                        delta = clampf(delta, -0.5f, 0.5f);
                        dval += delta;
                    }
                }
            }
            disp_out.at(x, y) = dval;
            if (best_cost) best_cost->at(x, y) = static_cast<float>(best);
            if (second_cost) second_cost->at(x, y) = static_cast<float>(second);
        }
    }
}

} // namespace apg
