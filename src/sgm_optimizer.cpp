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

inline void process_pixel(int x, int y, bool has_prev, int px, int py,
                          int D, int P1, const SgmParams& sgm_cfg,
                          const Image8& gray, const CostVolume& base, CostVolume32& acc,
                          std::vector<int>& prev, std::vector<int>& cur) {
    const uint16_t* c = base.slice(x, y);
    uint32_t* a = acc.slice(x, y);

    if (!has_prev) {
        for (int d = 0; d < D; ++d) {
            if (c[d] == kInvalidCost) {
                cur[d] = kPathInf;
                a[d] = kInvalidCost32;
            } else {
                cur[d] = static_cast<int>(c[d]);
                if (a[d] != kInvalidCost32) {
                    a[d] += static_cast<uint32_t>(cur[d]);
                }
            }
        }
        prev = cur;
        return;
    }

    const int dI = static_cast<int>(gray.at(x, y)) - static_cast<int>(gray.at(px, py));
    const int P2 = adaptive_p2(sgm_cfg, dI);

    int min_prev = kPathInf;
    for (int d = 0; d < D; ++d) {
        if (prev[d] < min_prev) min_prev = prev[d];
    }

    for (int d = 0; d < D; ++d) {
        const bool valid_cur = (c[d] != kInvalidCost);
        if (!valid_cur) {
            cur[d] = kPathInf;
            a[d] = kInvalidCost32;
            continue;
        }

        int best = kPathInf;
        if (prev[d] < kPathInf)
            best = std::min(best, prev[d]);
        if (d > 0 && prev[d - 1] < kPathInf)
            best = std::min(best, prev[d - 1] + P1);
        if (d + 1 < D && prev[d + 1] < kPathInf)
            best = std::min(best, prev[d + 1] + P1);
        if (min_prev < kPathInf)
            best = std::min(best, min_prev + P2);

        if (best >= kPathInf) {
            cur[d] = static_cast<int>(c[d]);
        } else {
            cur[d] = static_cast<int>(c[d]) + best - min_prev;
        }

        if (a[d] != kInvalidCost32) {
            a[d] += static_cast<uint32_t>(cur[d]);
        }
    }
    prev = cur;
}

} // namespace

void SgmOptimizer::aggregate_path(const PipelineConfig& cfg, const Image8& gray,
                                  const CostVolume& base, CostVolume32& acc, int dx, int dy) const {
    const int w = base.width();
    const int h = base.height();
    const int D = base.D();
    const int P1 = cfg.sgm.P1;

    std::vector<int> prev(D), cur(D);

    if (dx == 1 && dy == 0) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                process_pixel(x, y, x > 0, x - 1, y, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == -1 && dy == 0) {
        for (int y = 0; y < h; ++y) {
            for (int x = w - 1; x >= 0; --x) {
                process_pixel(x, y, x + 1 < w, x + 1, y, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == 1) {
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) {
                process_pixel(x, y, y > 0, x, y - 1, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == -1) {
        for (int x = 0; x < w; ++x) {
            for (int y = h - 1; y >= 0; --y) {
                process_pixel(x, y, y + 1 < h, x, y + 1, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
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
                    process_pixel(x, y, hp, px, py, D, P1, cfg.sgm, gray, base, acc, prev, cur);
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
    const int d0 = buf.cost.d0();
    buf.cost32.allocate(w, h, d0, d0 + D, 0);

    const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const int npath = (cfg.sgm.paths == PathType::Path8) ? 8 : 4;

    for (int p = 0; p < npath; ++p) {
        aggregate_path(cfg, buf.left_gray, buf.cost, buf.cost32, dirs[p][0], dirs[p][1]);
    }
}

namespace {

template <typename TCost>
void wta_impl(const PipelineConfig& cfg, const CostVolumeT<TCost>& vol,
              const SearchRange& range, Image32f& disp_out,
              Image32f* best_cost, Image32f* second_cost) {
    const int w = vol.width();
    const int h = vol.height();
    const int d0 = vol.d0();
    const int D = vol.D();
    disp_out = Image32f(w, h, -1.f);
    if (best_cost) *best_cost = Image32f(w, h, std::numeric_limits<float>::infinity());
    if (second_cost) *second_cost = Image32f(w, h, std::numeric_limits<float>::infinity());

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const TCost* s = vol.slice(x, y);
            const int lo = range.dmin.at(x, y);
            const int hi = range.dmax.at(x, y);
            if (hi <= lo) {
                disp_out.at(x, y) = -1.f;
                if (best_cost) best_cost->at(x, y) = std::numeric_limits<float>::infinity();
                if (second_cost) second_cost->at(x, y) = std::numeric_limits<float>::infinity();
                continue;
            }

            int best_d = -1;
            uint64_t best = UINT64_MAX;
            for (int d = lo; d < hi; ++d) {
                const int di = d - d0;
                if (di < 0 || di >= D) continue;
                const TCost c = s[di];
                if (c == static_cast<TCost>(kInvalidCost) ||
                    c == static_cast<TCost>(kInvalidCost32)) continue;
                if (static_cast<uint64_t>(c) < best) {
                    best = static_cast<uint64_t>(c);
                    best_d = d;
                }
            }

            if (best_d < 0) {
                disp_out.at(x, y) = -1.f;
                if (best_cost) best_cost->at(x, y) = std::numeric_limits<float>::infinity();
                if (second_cost) second_cost->at(x, y) = std::numeric_limits<float>::infinity();
                continue;
            }

            uint64_t second = UINT64_MAX;
            for (int d = lo; d < hi; ++d) {
                if (std::abs(d - best_d) <= 1) continue;
                const int di = d - d0;
                if (di < 0 || di >= D) continue;
                const TCost c = s[di];
                if (c == static_cast<TCost>(kInvalidCost) ||
                    c == static_cast<TCost>(kInvalidCost32)) continue;
                if (static_cast<uint64_t>(c) < second) {
                    second = static_cast<uint64_t>(c);
                }
            }

            float dval = static_cast<float>(best_d);
            if (cfg.post.subpixel && best_d - 1 >= lo && best_d + 1 < hi) {
                const int di = best_d - d0;
                if (di > 0 && di + 1 < D) {
                    const TCost c_m = s[di - 1];
                    const TCost c_0 = s[di];
                    const TCost c_p = s[di + 1];
                    if (c_m != static_cast<TCost>(kInvalidCost) &&
                        c_m != static_cast<TCost>(kInvalidCost32) &&
                        c_p != static_cast<TCost>(kInvalidCost) &&
                        c_p != static_cast<TCost>(kInvalidCost32)) {
                        const float cm = static_cast<float>(c_m);
                        const float c0 = static_cast<float>(c_0);
                        const float cp = static_cast<float>(c_p);
                        const float denom = cm - 2.f * c0 + cp;
                        if (std::abs(denom) > 1e-6f) {
                            float delta = 0.5f * (cm - cp) / denom;
                            delta = clampf(delta, -0.5f, 0.5f);
                            dval += delta;
                        }
                    }
                }
            }

            disp_out.at(x, y) = dval;
            if (best_cost) best_cost->at(x, y) = static_cast<float>(best);
            if (second_cost) {
                second_cost->at(x, y) = (second < UINT64_MAX)
                    ? static_cast<float>(second)
                    : std::numeric_limits<float>::infinity();
            }
        }
    }
}

} // namespace

void SgmOptimizer::winner_take_all(const PipelineConfig& cfg, const CostVolume32& vol,
                                   const SearchRange& range, Image32f& disp_out,
                                   Image32f* best_cost, Image32f* second_cost) const {
    wta_impl(cfg, vol, range, disp_out, best_cost, second_cost);
}

void SgmOptimizer::winner_take_all(const PipelineConfig& cfg, const CostVolume& vol,
                                   const SearchRange& range, Image32f& disp_out,
                                   Image32f* best_cost, Image32f* second_cost) const {
    wta_impl(cfg, vol, range, disp_out, best_cost, second_cost);
}

} // namespace apg
