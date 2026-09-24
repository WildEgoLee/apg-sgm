#pragma once

#include "apg_sgm/image.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>

namespace apg {

struct SearchRange {
    Image16s dmin;
    Image16s dmax;
    int global_min = 0;
    int global_max = 0;

    int D() const { return global_max - global_min; }

    void allocate(int w, int h, int gmin, int gmax) {
        global_min = gmin;
        global_max = gmax;
        dmin = Image16s(w, h, static_cast<int16_t>(gmin));
        dmax = Image16s(w, h, static_cast<int16_t>(gmax));
    }

    bool valid(int x, int y) const {
        return dmax.at(x, y) > dmin.at(x, y);
    }

    void set_pixel(int x, int y, int lo, int hi) {
        lo = std::max(lo, global_min);
        hi = std::min(hi, global_max);
        if (hi < lo) hi = lo;
        dmin.at(x, y) = static_cast<int16_t>(lo);
        dmax.at(x, y) = static_cast<int16_t>(hi);
    }
};

} // namespace apg
