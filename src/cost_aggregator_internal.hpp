#pragma once

#include "apg_sgm/packed_volume.hpp"

#include <memory>
#include <cstdint>
#include <utility>

namespace apg {
namespace detail {

// Uninitialized for-overwrite workspace for Cross aggregation tmp buffer.
// Eliminates redundant zero/fill initialization overhead across hundreds of MiB / GiB of memory.
//
// Key Invariants:
// 1. tmp lifetime remains strictly confined to CostAggregator::aggregate_packed().
// 2. No persistent full-volume workspace is retained across frames or phases, preserving the <= 3*C16 peak model.
// 3. The horizontal prefix pass unconditionally overwrites every valid packed disparity state in tmp
//    before any vertical pass read, guaranteeing memory correctness with uninitialized storage.
struct PackedCrossTmp {
    std::shared_ptr<const PackedVolumeLayout> layout_;
    std::unique_ptr<uint16_t[]> data_;

    explicit PackedCrossTmp(std::shared_ptr<const PackedVolumeLayout> l)
        : layout_(std::move(l)),
          data_(layout_ && layout_->state_count() > 0 ? new uint16_t[layout_->state_count()] : nullptr) {}

    int width() const { return layout_ ? layout_->width : 0; }
    int height() const { return layout_ ? layout_->height : 0; }
    int dmin(int x, int y) const {
        return layout_ ? layout_->dmin[static_cast<size_t>(y) * layout_->width + x] : 0;
    }
    int dmax(int x, int y) const {
        return layout_ ? layout_->dmax(static_cast<size_t>(y) * layout_->width + x) : 0;
    }
    uint16_t* slice(int x, int y) {
        return data_.get() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }
    const uint16_t* slice(int x, int y) const {
        return data_.get() + layout_->offsets[static_cast<size_t>(y) * layout_->width + x];
    }
};

} // namespace detail
} // namespace apg
