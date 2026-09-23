#pragma once

#include "apg_sgm/image.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>
#include <vector>

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

    void set_pixel(int x, int y, int lo, int hi) {
        lo = std::max(lo, global_min);
        hi = std::min(hi, global_max);
        if (hi <= lo) hi = std::min(lo + 1, global_max);
        dmin.at(x, y) = static_cast<int16_t>(lo);
        dmax.at(x, y) = static_cast<int16_t>(hi);
    }
};

class CostVolume {
public:
    CostVolume() = default;
    CostVolume(int w, int h, int d0, int d1, uint16_t fill = 0) { allocate(w, h, d0, d1, fill); }

    void allocate(int w, int h, int d0, int d1, uint16_t fill = 0) {
        w_ = w;
        h_ = h;
        d0_ = d0;
        D_ = d1 - d0;
        data_.assign(static_cast<size_t>(w) * h * D_, fill);
    }

    int width() const { return w_; }
    int height() const { return h_; }
    int d0() const { return d0_; }
    int D() const { return D_; }
    bool empty() const { return data_.empty(); }

    uint16_t* data() { return data_.data(); }
    const uint16_t* data() const { return data_.data(); }

    size_t index(int x, int y, int d) const {
        return (static_cast<size_t>(y) * w_ + x) * D_ + (d - d0_);
    }

    uint16_t& at(int x, int y, int d) { return data_[index(x, y, d)]; }
    uint16_t at(int x, int y, int d) const { return data_[index(x, y, d)]; }

    uint16_t* slice(int x, int y) { return data_.data() + (static_cast<size_t>(y) * w_ + x) * D_; }
    const uint16_t* slice(int x, int y) const {
        return data_.data() + (static_cast<size_t>(y) * w_ + x) * D_;
    }

    size_t bytes() const { return data_.size() * sizeof(uint16_t); }

private:
    int w_ = 0, h_ = 0, d0_ = 0, D_ = 0;
    std::vector<uint16_t> data_;
};

struct PipelineBuffers {
    Image8 left;
    Image8 right;
    Image8 left_gray;
    Image8 right_gray;
    Image8 left_gx, left_gy, right_gx, right_gy;

    std::vector<uint32_t> census_left;
    std::vector<uint32_t> census_right;
    std::vector<uint64_t> census_left64;
    std::vector<uint64_t> census_right64;

    SearchRange range;
    Image32f d_prior;
    CostVolume cost;
    CostVolume cost_right;
    Image32f disparity;
    Image32f disparity_right;
    Image32f confidence;
    Image8u1 invalid_reason;
    Image8u1 reliable_mask;
};

} // namespace apg
