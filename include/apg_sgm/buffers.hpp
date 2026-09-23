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

template <typename T>
class CostVolumeT {
public:
    CostVolumeT() = default;
    CostVolumeT(int w, int h, int d0, int d1, T fill = 0) { allocate(w, h, d0, d1, fill); }

    void allocate(int w, int h, int d0, int d1, T fill = 0) {
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

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    size_t index(int x, int y, int d) const {
        return (static_cast<size_t>(y) * w_ + x) * D_ + (d - d0_);
    }

    T& at(int x, int y, int d) { return data_[index(x, y, d)]; }
    T at(int x, int y, int d) const { return data_[index(x, y, d)]; }

    T* slice(int x, int y) { return data_.data() + (static_cast<size_t>(y) * w_ + x) * D_; }
    const T* slice(int x, int y) const {
        return data_.data() + (static_cast<size_t>(y) * w_ + x) * D_;
    }

    size_t bytes() const { return data_.size() * sizeof(T); }

private:
    int w_ = 0, h_ = 0, d0_ = 0, D_ = 0;
    std::vector<T> data_;
};

using CostVolume = CostVolumeT<uint16_t>;
using CostVolume32 = CostVolumeT<uint32_t>;

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
    size_t support_count = 0;
    Image32f d_prior;
    Image32f prior_confidence;
    Image32f prior_spread;
    CostVolume cost;
    CostVolume32 cost32;
    CostVolume cost_right;
    Image32f disparity;
    Image32f d_before_refine;
    Image32f disparity_right;
    Image32f confidence;
    Image8u1 invalid_reason;
    Image8u1 reliable_mask;
};

} // namespace apg
