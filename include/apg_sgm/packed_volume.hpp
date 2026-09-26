#pragma once

#include "apg_sgm/search_range.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>
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

    PackedCostVolume(const PackedCostVolume& other)
        : layout_(other.layout_), size_(other.data_ ? other.size_ : 0) {
        if (size_ > 0) {
            data_.reset(new T[size_]);
            std::copy_n(other.data_.get(), size_, data_.get());
        }
    }

    PackedCostVolume& operator=(const PackedCostVolume& other) {
        if (this != &other) {
            const size_t new_size = other.data_ ? other.size_ : 0;
            std::unique_ptr<T[]> new_data;
            if (new_size > 0) {
                new_data.reset(new T[new_size]);
                std::copy_n(other.data_.get(), new_size, new_data.get());
            }
            layout_ = other.layout_;
            size_ = new_size;
            data_ = std::move(new_data);
        }
        return *this;
    }

    PackedCostVolume(PackedCostVolume&& other) noexcept
        : layout_(std::move(other.layout_)),
          size_(other.size_),
          data_(std::move(other.data_)) {
        other.size_ = 0;
    }

    PackedCostVolume& operator=(PackedCostVolume&& other) noexcept {
        if (this != &other) {
            layout_ = std::move(other.layout_);
            size_ = other.size_;
            data_ = std::move(other.data_);
            other.size_ = 0;
        }
        return *this;
    }

    explicit PackedCostVolume(std::shared_ptr<const PackedVolumeLayout> layout, T fill = 0) {
        allocate(std::move(layout), fill);
    }

    explicit PackedCostVolume(const SearchRange& range, T fill = 0) {
        allocate(range, fill);
    }

    void allocate_for_overwrite(std::shared_ptr<const PackedVolumeLayout> layout) {
        const size_t new_size = layout ? layout->state_count() : 0;
        std::unique_ptr<T[]> new_data;
        if (new_size > 0) {
            new_data.reset(new T[new_size]);
        }
        layout_ = std::move(layout);
        size_ = new_size;
        data_ = std::move(new_data);
    }

    void allocate_for_overwrite(const SearchRange& range) {
        allocate_for_overwrite(PackedVolumeLayout::from_range(range));
    }

    void allocate(std::shared_ptr<const PackedVolumeLayout> layout, T fill = 0) {
        allocate_for_overwrite(std::move(layout));
        if (size_ > 0) {
            std::fill_n(data_.get(), size_, fill);
        }
    }

    void allocate(const SearchRange& range, T fill = 0) {
        allocate(PackedVolumeLayout::from_range(range), fill);
    }

    void fill(T value) {
        if (size_ > 0) {
            std::fill_n(data_.get(), size_, value);
        }
    }

    void release() {
        data_.reset();
        size_ = 0;
        layout_.reset();
    }

    std::shared_ptr<const PackedVolumeLayout> layout() const { return layout_; }

    int width() const { return layout_ ? layout_->width : 0; }
    int height() const { return layout_ ? layout_->height : 0; }
    bool empty() const { return size_ == 0; }
    size_t total_elements() const { return size_; }

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
        return data_.get() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }

    const T* slice(int x, int y) const {
        return data_.get() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
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

    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }

    size_t data_bytes() const { return size_ * sizeof(T); }
    size_t layout_bytes() const { return layout_ ? layout_->bytes() : 0; }
    size_t bytes() const { return data_bytes() + layout_bytes(); }

private:
    std::shared_ptr<const PackedVolumeLayout> layout_;
    size_t size_ = 0;
    std::unique_ptr<T[]> data_;
};

using PackedCostVolume16 = PackedCostVolume<uint16_t>;
using PackedCostVolume32 = PackedCostVolume<uint32_t>;

} // namespace apg
