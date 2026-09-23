#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace apg {

struct PackedVolumeLayout {
    int width = 0;
    int height = 0;
    std::vector<uint32_t> offsets; // N + 1 prefix-sum offsets
    std::vector<int16_t> dmin;     // N per-pixel dmin

    static std::shared_ptr<PackedVolumeLayout> from_range(const SearchRange& range) {
        auto layout = std::make_shared<PackedVolumeLayout>();
        layout->width = range.dmin.width();
        layout->height = range.dmin.height();
        const size_t n_pixels = static_cast<size_t>(layout->width) * layout->height;
        layout->offsets.resize(n_pixels + 1);
        layout->dmin.resize(n_pixels);

        uint64_t total_elements = 0;
        layout->offsets[0] = 0;

        for (int y = 0; y < layout->height; ++y) {
            for (int x = 0; x < layout->width; ++x) {
                const size_t p = static_cast<size_t>(y) * layout->width + x;
                const int d0 = range.dmin.at(x, y);
                const int d1 = range.dmax.at(x, y);
                const int D_p = (d1 > d0) ? (d1 - d0) : 0;
                total_elements += static_cast<uint64_t>(D_p);
                if (total_elements > UINT32_MAX) {
                    throw std::runtime_error("PackedVolumeLayout: total disparity states exceed uint32 capacity");
                }
                layout->offsets[p + 1] = static_cast<uint32_t>(total_elements);
                layout->dmin[p] = static_cast<int16_t>(d0);
            }
        }
        return layout;
    }

    uint32_t state_count() const {
        return offsets.empty() ? 0 : offsets.back();
    }

    int disp_width(size_t p) const {
        return static_cast<int>(offsets[p + 1] - offsets[p]);
    }

    int dmax(size_t p) const {
        return static_cast<int>(dmin[p]) + disp_width(p);
    }

    size_t bytes() const {
        return offsets.size() * sizeof(uint32_t) + dmin.size() * sizeof(int16_t);
    }
};

template <typename T>
class PackedCostVolume {
public:
    PackedCostVolume() = default;

    explicit PackedCostVolume(std::shared_ptr<const PackedVolumeLayout> layout, T fill = 0) {
        allocate(std::move(layout), fill);
    }

    explicit PackedCostVolume(const SearchRange& range, T fill = 0) {
        allocate(range, fill);
    }

    void allocate(std::shared_ptr<const PackedVolumeLayout> layout, T fill = 0) {
        layout_ = std::move(layout);
        if (layout_) {
            data_.assign(layout_->state_count(), fill);
        } else {
            data_.clear();
        }
    }

    void allocate(const SearchRange& range, T fill = 0) {
        allocate(PackedVolumeLayout::from_range(range), fill);
    }

    std::shared_ptr<const PackedVolumeLayout> layout() const { return layout_; }

    int width() const { return layout_ ? layout_->width : 0; }
    int height() const { return layout_ ? layout_->height : 0; }
    bool empty() const { return data_.empty(); }
    size_t total_elements() const { return data_.size(); }

    int dmin(int x, int y) const {
        return layout_ ? layout_->dmin[static_cast<size_t>(y) * layout_->width + x] : 0;
    }
    int dmax(int x, int y) const {
        return layout_ ? layout_->dmax(static_cast<size_t>(y) * layout_->width + x) : 0;
    }
    int disp_width(int x, int y) const {
        return layout_ ? layout_->disp_width(static_cast<size_t>(y) * layout_->width + x) : 0;
    }

    bool contains(int x, int y, int d) const {
        if (!layout_ || x < 0 || x >= layout_->width || y < 0 || y >= layout_->height) return false;
        const size_t p = static_cast<size_t>(y) * layout_->width + x;
        return d >= layout_->dmin[p] && d < layout_->dmax(p);
    }

    uint32_t offset(int x, int y) const {
        return layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }

    T* slice(int x, int y) {
        return data_.data() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }

    const T* slice(int x, int y) const {
        return data_.data() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }

    T& at(int x, int y, int d) {
        const size_t p = static_cast<size_t>(y) * layout_->width + x;
        const size_t idx = layout_->offsets[p] + static_cast<size_t>(d - layout_->dmin[p]);
        return data_[idx];
    }

    const T& at(int x, int y, int d) const {
        const size_t p = static_cast<size_t>(y) * layout_->width + x;
        const size_t idx = layout_->offsets[p] + static_cast<size_t>(d - layout_->dmin[p]);
        return data_[idx];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    size_t data_bytes() const { return data_.size() * sizeof(T); }
    size_t layout_bytes() const { return layout_ ? layout_->bytes() : 0; }
    size_t bytes() const { return data_bytes() + layout_bytes(); }

private:
    std::shared_ptr<const PackedVolumeLayout> layout_;
    std::vector<T> data_;
};

using PackedCostVolume16 = PackedCostVolume<uint16_t>;
using PackedCostVolume32 = PackedCostVolume<uint32_t>;

} // namespace apg
