#pragma once

#include "apg_sgm/image.hpp"

namespace apg {

struct StereoResult {
    Image32f disparity;
    Image32f confidence;
    Image8u1 invalid_reason;
    Image8u1 reliable_mask;

    void clear() {
        disparity = Image32f();
        confidence = Image32f();
        invalid_reason = Image8u1();
        reliable_mask = Image8u1();
    }

    bool empty() const {
        return disparity.empty();
    }

    int width() const {
        return disparity.width();
    }

    int height() const {
        return disparity.height();
    }
};

} // namespace apg
