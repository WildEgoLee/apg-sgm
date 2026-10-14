#pragma once

#include "apg_sgm/image.hpp"
#include "apg_sgm/packed_volume.hpp"
#include "apg_sgm/search_range.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace apg {

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

    void release() {
        data_.clear();
        data_.shrink_to_fit();
        w_ = 0;
        h_ = 0;
        d0_ = 0;
        D_ = 0;
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
    size_t capacity_bytes() const { return data_.capacity() * sizeof(T); }

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
    std::vector<SupportMatch> supports;
    size_t support_count = 0;
    Image32f d_prior;
    Image32f prior_confidence;
    Image32f prior_spread;
    Image32f d_prior_min;
    Image32f d_prior_max;
    CostVolume cost;
    CostVolume32 cost32;
    CostVolume cost_right;
    PackedCostVolume16 packed_cost;
    PackedCostVolume16 packed_sgm16;
    PackedCostVolume32 packed_cost32;
    Image32f disparity;
    Image32f d_before_refine;
    Image32f d_after_refine;
    Image32f disparity_right;
    Image32f confidence;
    Image8u1 invalid_reason;
    Image8u1 reliable_mask;

    std::shared_ptr<PipelineBuffers> coarse_buffers;

    void release();
    size_t allocated_bytes() const;
};

inline void PipelineBuffers::release() {
    left.release();
    right.release();
    left_gray.release();
    right_gray.release();
    left_gx.release();
    left_gy.release();
    right_gx.release();
    right_gy.release();

    census_left.clear(); census_left.shrink_to_fit();
    census_right.clear(); census_right.shrink_to_fit();
    census_left64.clear(); census_left64.shrink_to_fit();
    census_right64.clear(); census_right64.shrink_to_fit();

    range.release();
    supports.clear(); supports.shrink_to_fit();
    support_count = 0;
    d_prior.release();
    prior_confidence.release();
    prior_spread.release();
    d_prior_min.release();
    d_prior_max.release();
    cost.release();
    cost32.release();
    cost_right.release();
    packed_cost.release();
    packed_sgm16.release();
    packed_cost32.release();
    disparity.release();
    d_before_refine.release();
    d_after_refine.release();
    disparity_right.release();
    confidence.release();
    invalid_reason.release();
    reliable_mask.release();
    if (coarse_buffers) coarse_buffers->release();
}

inline size_t PipelineBuffers::allocated_bytes() const {
    size_t total = 0;
    total += left.bytes() + right.bytes();
    total += left_gray.bytes() + right_gray.bytes();
    total += left_gx.bytes() + left_gy.bytes() + right_gx.bytes() + right_gy.bytes();
    total += census_left.capacity() * sizeof(uint32_t);
    total += census_right.capacity() * sizeof(uint32_t);
    total += census_left64.capacity() * sizeof(uint64_t);
    total += census_right64.capacity() * sizeof(uint64_t);
    total += range.bytes();
    total += supports.capacity() * sizeof(SupportMatch);
    total += d_prior.bytes() + prior_confidence.bytes() + prior_spread.bytes();
    total += d_prior_min.bytes() + d_prior_max.bytes();
    total += cost.capacity_bytes() + cost32.capacity_bytes() + cost_right.capacity_bytes();
    total += packed_cost.allocated_bytes() + packed_sgm16.allocated_bytes() + packed_cost32.allocated_bytes();
    total += disparity.bytes() + d_before_refine.bytes() + d_after_refine.bytes();
    total += disparity_right.bytes() + confidence.bytes();
    total += invalid_reason.bytes() + reliable_mask.bytes();
    if (coarse_buffers) total += coarse_buffers->allocated_bytes();
    return total;
}

} // namespace apg
