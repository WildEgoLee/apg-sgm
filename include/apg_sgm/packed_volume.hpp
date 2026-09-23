#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace apg {

template <typename T>
class PackedCostVolume {
public:
    PackedCostVolume() = default;

    void allocate(const SearchRange& range, T fill = 0) {
        range_ = &range;
        w_ = range.dmin.width();
        h_ = range.dmin.height();
        const size_t n_pixels = static_cast<size_t>(w_) * h_;
        offsets_.resize(n_pixels + 1);

        size_t total_elements = 0;
        offsets_[0] = 0;
        for (int y = 0; y < h_; ++y) {
            for (int x = 0; x < w_; ++x) {
                const size_t p = static_cast<size_t>(y) * w_ + x;
                const int d0 = range.dmin.at(x, y);
                const int d1 = range.dmax.at(x, y);
                const int D_p = (d1 > d0) ? (d1 - d0) : 0;
                total_elements += D_p;
                offsets_[p + 1] = total_elements;
            }
        }

        data_.assign(total_elements, fill);
    }

    int width() const { return w_; }
    int height() const { return h_; }
    bool empty() const { return data_.empty(); }
    size_t total_elements() const { return data_.size(); }

    int dmin(int x, int y) const { return range_ ? range_->dmin.at(x, y) : 0; }
    int dmax(int x, int y) const { return range_ ? range_->dmax.at(x, y) : 0; }
    int disp_width(int x, int y) const {
        if (!range_) return 0;
        const int d0 = range_->dmin.at(x, y);
        const int d1 = range_->dmax.at(x, y);
        return (d1 > d0) ? (d1 - d0) : 0;
    }

    bool contains(int x, int y, int d) const {
        if (!range_ || x < 0 || x >= w_ || y < 0 || y >= h_) return false;
        return d >= range_->dmin.at(x, y) && d < range_->dmax.at(x, y);
    }

    size_t offset(int x, int y) const {
        return offsets_[static_cast<size_t>(y) * w_ + x];
    }

    T* slice(int x, int y) {
        return data_.data() + offsets_[static_cast<size_t>(y) * w_ + x];
    }

    const T* slice(int x, int y) const {
        return data_.data() + offsets_[static_cast<size_t>(y) * w_ + x];
    }

    T& at(int x, int y, int d) {
        const int d0 = range_->dmin.at(x, y);
        const size_t idx = offsets_[static_cast<size_t>(y) * w_ + x] + static_cast<size_t>(d - d0);
        return data_[idx];
    }

    const T& at(int x, int y, int d) const {
        const int d0 = range_->dmin.at(x, y);
        const size_t idx = offsets_[static_cast<size_t>(y) * w_ + x] + static_cast<size_t>(d - d0);
        return data_[idx];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    const std::vector<size_t>& offsets() const { return offsets_; }

    size_t data_bytes() const { return data_.size() * sizeof(T); }
    size_t offsets_bytes() const { return offsets_.size() * sizeof(size_t); }
    size_t bytes() const { return data_bytes() + offsets_bytes(); }

private:
    const SearchRange* range_ = nullptr;
    int w_ = 0;
    int h_ = 0;
    std::vector<size_t> offsets_;
    std::vector<T> data_;
};

using PackedCostVolume16 = PackedCostVolume<uint16_t>;
using PackedCostVolume32 = PackedCostVolume<uint32_t>;

} // namespace apg
