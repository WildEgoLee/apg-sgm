#pragma once

#include "apg_sgm/types.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace apg {

class Image8 {
public:
    Image8() = default;
    Image8(int w, int h, int channels = 1);

    int width() const { return w_; }
    int height() const { return h_; }
    int channels() const { return c_; }
    Size size() const { return {w_, h_}; }
    bool empty() const { return data_.empty(); }

    uint8_t* data() { return data_.data(); }
    const uint8_t* data() const { return data_.data(); }

    uint8_t& at(int x, int y, int ch = 0) {
        return data_[(y * w_ + x) * c_ + ch];
    }
    uint8_t at(int x, int y, int ch = 0) const {
        return data_[(y * w_ + x) * c_ + ch];
    }

    uint8_t sample(int x, int y, int ch = 0) const;
    uint8_t gray(int x, int y) const;
    void fill(uint8_t v);

private:
    int w_ = 0;
    int h_ = 0;
    int c_ = 1;
    std::vector<uint8_t> data_;
};

class Image32f {
public:
    Image32f() = default;
    Image32f(int w, int h, float fill = 0.f);

    int width() const { return w_; }
    int height() const { return h_; }
    Size size() const { return {w_, h_}; }
    bool empty() const { return data_.empty(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    float& at(int x, int y) { return data_[y * w_ + x]; }
    float at(int x, int y) const { return data_[y * w_ + x]; }
    float& operator[](int i) { return data_[i]; }
    float operator[](int i) const { return data_[i]; }

    void fill(float v);

private:
    int w_ = 0;
    int h_ = 0;
    std::vector<float> data_;
};

class Image16s {
public:
    Image16s() = default;
    Image16s(int w, int h, int16_t fill = 0);

    int width() const { return w_; }
    int height() const { return h_; }
    bool empty() const { return data_.empty(); }
    int16_t* data() { return data_.data(); }
    const int16_t* data() const { return data_.data(); }
    int16_t& at(int x, int y) { return data_[y * w_ + x]; }
    int16_t at(int x, int y) const { return data_[y * w_ + x]; }
    void fill(int16_t v);

private:
    int w_ = 0;
    int h_ = 0;
    std::vector<int16_t> data_;
};

class Image8u1 {
public:
    Image8u1() = default;
    Image8u1(int w, int h, uint8_t fill = 0);

    int width() const { return w_; }
    int height() const { return h_; }
    bool empty() const { return data_.empty(); }
    uint8_t* data() { return data_.data(); }
    const uint8_t* data() const { return data_.data(); }
    uint8_t& at(int x, int y) { return data_[y * w_ + x]; }
    uint8_t at(int x, int y) const { return data_[y * w_ + x]; }
    void fill(uint8_t v);

private:
    int w_ = 0;
    int h_ = 0;
    std::vector<uint8_t> data_;
};

bool load_image(const std::string& path, Image8& out);
bool save_pgm(const std::string& path, const Image8& img);
bool save_pgm16(const std::string& path, const Image16s& disp, int scale = kSubpixelScale);
bool save_disparity_preview(const std::string& path, const Image32f& disp, float dmax);
bool load_pfm(const std::string& path, Image32f& out);
bool save_pfm(const std::string& path, const Image32f& img);

} // namespace apg
