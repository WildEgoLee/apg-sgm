#pragma once

#include "apg_sgm/config.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace apg {
namespace detail {

struct PathState {
    int dmin = 0;
    int dmax = 0;
    int min_val = kPathInf;
    std::vector<int> vals;
};

inline int adaptive_p2(const SgmParams& p, int dI) {
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

} // namespace detail
} // namespace apg
