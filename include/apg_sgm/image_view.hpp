#pragma once

#include "apg_sgm/image.hpp"
#include "apg_sgm/types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace apg {

class ImageView8 {
public:
    ImageView8() = default;

    ImageView8(const uint8_t* data, int width, int height, int stride_bytes, int channels = 1)
        : data_(data), w_(width), h_(height), stride_(stride_bytes), c_(channels) {}

    ImageView8(const Image8& img)
        : data_(img.data()),
          w_(img.width()),
          h_(img.height()),
          stride_(img.width() * img.channels()),
          c_(img.channels()) {}

    int width() const { return w_; }
    int height() const { return h_; }
    int stride() const { return stride_; }
    int channels() const { return c_; }
    Size size() const { return {w_, h_}; }
    bool empty() const { return data_ == nullptr || w_ <= 0 || h_ <= 0; }
    const uint8_t* data() const { return data_; }

    const uint8_t* row(int y) const {
        return data_ + static_cast<size_t>(y) * stride_;
    }

    uint8_t at(int x, int y, int ch = 0) const {
        return row(y)[x * c_ + ch];
    }

    uint8_t sample(int x, int y, int ch = 0) const {
        x = clampi(x, 0, w_ - 1);
        y = clampi(y, 0, h_ - 1);
        return at(x, y, ch);
    }

    uint8_t gray(int x, int y) const {
        if (c_ == 1) return at(x, y, 0);
        const int r = at(x, y, 0);
        const int g = at(x, y, 1);
        const int b = at(x, y, 2);
        return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
    }

    Image8 to_image() const {
        if (empty()) return Image8();
        Image8 out(w_, h_, c_);
        for (int y = 0; y < h_; ++y) {
            std::memcpy(out.data() + static_cast<size_t>(y) * w_ * c_,
                        row(y), static_cast<size_t>(w_) * c_);
        }
        return out;
    }

private:
    const uint8_t* data_ = nullptr;
    int w_ = 0;
    int h_ = 0;
    int stride_ = 0;
    int c_ = 1;
};

struct StereoInputView {
    ImageView8 left;
    ImageView8 right;

    StereoInputView() = default;
    StereoInputView(ImageView8 l, ImageView8 r) : left(l), right(r) {}
    StereoInputView(const Image8& l, const Image8& r) : left(l), right(r) {}
};

} // namespace apg
