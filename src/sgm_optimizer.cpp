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

struct PathState {
    int dmin = 0;
    int dmax = 0;
    int min_val = kPathInf;
    std::vector<int> vals;
};

inline void process_pixel_packed(int x, int y, bool has_prev, int px, int py,
                                 int P1, const SgmParams& sgm_cfg,
                                 const Image8& gray, const PackedCostVolume16& base,
                                 PackedCostVolume32& acc,
                                 PathState& prev, PathState& cur) {
    const int dmin_c = base.dmin(x, y);
    const int dmax_c = base.dmax(x, y);
    const int D_c = dmax_c - dmin_c;
    cur.dmin = dmin_c;
    cur.dmax = dmax_c;
    cur.vals.resize(D_c);
    cur.min_val = kPathInf;

    if (D_c <= 0) {
        prev = cur;
        return;
    }

    const uint16_t* c = base.slice(x, y);
    uint32_t* a = acc.slice(x, y);

    if (!has_prev || prev.vals.empty()) {
        for (int di = 0; di < D_c; ++di) {
            if (c[di] == kInvalidCost) {
                cur.vals[di] = kPathInf;
                a[di] = kInvalidCost32;
            } else {
                cur.vals[di] = static_cast<int>(c[di]);
                if (cur.vals[di] < cur.min_val) {
                    cur.min_val = cur.vals[di];
                }
                if (a[di] != kInvalidCost32) {
                    a[di] += static_cast<uint32_t>(cur.vals[di]);
                }
            }
        }
        prev = cur;
        return;
    }

    const int dI = static_cast<int>(gray.at(x, y)) - static_cast<int>(gray.at(px, py));
    const int P2 = adaptive_p2(sgm_cfg, dI);

    const int dmin_p = prev.dmin;
    const int dmax_p = prev.dmax;
    const int min_prev = prev.min_val;
    const auto& prev_vals = prev.vals;

    for (int di = 0; di < D_c; ++di) {
        const bool valid_cur = (c[di] != kInvalidCost);
        if (!valid_cur) {
            cur.vals[di] = kPathInf;
            a[di] = kInvalidCost32;
            continue;
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

        if (a[di] != kInvalidCost32) {
            a[di] += static_cast<uint32_t>(cur.vals[di]);
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

    if (dx == 1 && dy == 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            std::vector<int> prev(D), cur(D);
            for (int x = 0; x < w; ++x) {
                process_pixel(x, y, x > 0, x - 1, y, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == -1 && dy == 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            std::vector<int> prev(D), cur(D);
            for (int x = w - 1; x >= 0; --x) {
                process_pixel(x, y, x + 1 < w, x + 1, y, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int x = 0; x < w; ++x) {
            std::vector<int> prev(D), cur(D);
            for (int y = 0; y < h; ++y) {
                process_pixel(x, y, y > 0, x, y - 1, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == -1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int x = 0; x < w; ++x) {
            std::vector<int> prev(D), cur(D);
            for (int y = h - 1; y >= 0; --y) {
                process_pixel(x, y, y + 1 < h, x, y + 1, D, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else {
        std::vector<int> prev(D), cur(D);
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

void SgmOptimizer::aggregate_path_packed(const PipelineConfig& cfg, const Image8& gray,
                                         const PackedCostVolume16& base,
                                         PackedCostVolume32& acc, int dx, int dy) const {
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
                process_pixel_packed(x, y, x > 0, x - 1, y, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == -1 && dy == 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int y = 0; y < h; ++y) {
            PathState prev, cur;
            for (int x = w - 1; x >= 0; --x) {
                process_pixel_packed(x, y, x + 1 < w, x + 1, y, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int x = 0; x < w; ++x) {
            PathState prev, cur;
            for (int y = 0; y < h; ++y) {
                process_pixel_packed(x, y, y > 0, x, y - 1, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else if (dx == 0 && dy == -1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (int x = 0; x < w; ++x) {
            PathState prev, cur;
            for (int y = h - 1; y >= 0; --y) {
                process_pixel_packed(x, y, y + 1 < h, x, y + 1, P1, cfg.sgm, gray, base, acc, prev, cur);
            }
        }
    } else {
        std::vector<uint8_t> seen(static_cast<size_t>(w) * h, 0);
        auto inb = [&](int x, int y) { return x >= 0 && x < w && y >= 0 && y < h; };
        PathState prev, cur;
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
                    process_pixel_packed(x, y, hp, px, py, P1, cfg.sgm, gray, base, acc, prev, cur);
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

void SgmOptimizer::optimize_packed(const PipelineConfig& cfg, const Image8& gray,
                                   const PackedCostVolume16& packed_cost,
                                   PackedCostVolume32& packed_cost32) const {
    if (!packed_cost32.layout() || packed_cost32.layout() != packed_cost.layout()) {
        packed_cost32.allocate(packed_cost.layout(), 0);
    } else {
        packed_cost32.fill(0);
    }

    const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const int npath = (cfg.sgm.paths == PathType::Path8) ? 8 : 4;

    for (int p = 0; p < npath; ++p) {
        aggregate_path_packed(cfg, gray, packed_cost, packed_cost32, dirs[p][0], dirs[p][1]);
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

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
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

            const TCost inv_c = invalid_cost<TCost>();
            int best_d = -1;
            uint64_t best = UINT64_MAX;
            for (int d = lo; d < hi; ++d) {
                const int di = d - d0;
                if (di < 0 || di >= D) continue;
                const TCost c = s[di];
                if (c == inv_c) continue;
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
                if (c == inv_c) continue;
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
                    if (c_m != inv_c && c_p != inv_c) {
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

template <typename TCost>
void wta_packed_impl(const PipelineConfig& cfg, const PackedCostVolume<TCost>& vol,
                     Image32f& disp_out, Image32f* best_cost, Image32f* second_cost) {
    const int w = vol.width();
    const int h = vol.height();
    disp_out = Image32f(w, h, -1.f);
    if (best_cost) *best_cost = Image32f(w, h, std::numeric_limits<float>::infinity());
    if (second_cost) *second_cost = Image32f(w, h, std::numeric_limits<float>::infinity());

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int lo = vol.dmin(x, y);
            const int hi = vol.dmax(x, y);
            const int D_p = hi - lo;
            if (D_p <= 0) {
                disp_out.at(x, y) = -1.f;
                if (best_cost) best_cost->at(x, y) = std::numeric_limits<float>::infinity();
                if (second_cost) second_cost->at(x, y) = std::numeric_limits<float>::infinity();
                continue;
            }

            const TCost* s = vol.slice(x, y);
            const TCost inv_c = invalid_cost<TCost>();
            int best_d = -1;
            uint64_t best = UINT64_MAX;

            for (int di = 0; di < D_p; ++di) {
                const TCost c = s[di];
                if (c == inv_c) continue;
                if (static_cast<uint64_t>(c) < best) {
                    best = static_cast<uint64_t>(c);
                    best_d = lo + di;
                }
            }

            if (best_d < 0) {
                disp_out.at(x, y) = -1.f;
                if (best_cost) best_cost->at(x, y) = std::numeric_limits<float>::infinity();
                if (second_cost) second_cost->at(x, y) = std::numeric_limits<float>::infinity();
                continue;
            }

            uint64_t second = UINT64_MAX;
            for (int di = 0; di < D_p; ++di) {
                const int d = lo + di;
                if (std::abs(d - best_d) <= 1) continue;
                const TCost c = s[di];
                if (c == inv_c) continue;
                if (static_cast<uint64_t>(c) < second) {
                    second = static_cast<uint64_t>(c);
                }
            }

            float dval = static_cast<float>(best_d);
            if (cfg.post.subpixel && best_d - 1 >= lo && best_d + 1 < hi) {
                const int di = best_d - lo;
                if (di > 0 && di + 1 < D_p) {
                    const TCost c_m = s[di - 1];
                    const TCost c_0 = s[di];
                    const TCost c_p = s[di + 1];
                    if (c_m != inv_c && c_p != inv_c) {
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

void SgmOptimizer::winner_take_all_packed(const PipelineConfig& cfg,
                                          const PackedCostVolume32& vol,
                                          Image32f& disp_out,
                                          Image32f* best_cost,
                                          Image32f* second_cost) const {
    wta_packed_impl(cfg, vol, disp_out, best_cost, second_cost);
}

void SgmOptimizer::winner_take_all_packed(const PipelineConfig& cfg,
                                          const PackedCostVolume16& vol,
                                          Image32f& disp_out,
                                          Image32f* best_cost,
                                          Image32f* second_cost) const {
    wta_packed_impl(cfg, vol, disp_out, best_cost, second_cost);
}

} // namespace apg
