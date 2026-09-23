#include "apg_sgm/image.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace apg {

Image8::Image8(int w, int h, int channels)
    : w_(w), h_(h), c_(channels), data_(static_cast<size_t>(w) * h * channels, 0) {}

uint8_t Image8::sample(int x, int y, int ch) const {
    x = clampi(x, 0, w_ - 1);
    y = clampi(y, 0, h_ - 1);
    return at(x, y, ch);
}

uint8_t Image8::gray(int x, int y) const {
    if (c_ == 1) return at(x, y, 0);
    const int r = at(x, y, 0);
    const int g = at(x, y, 1);
    const int b = at(x, y, 2);
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

void Image8::fill(uint8_t v) { std::fill(data_.begin(), data_.end(), v); }

Image32f::Image32f(int w, int h, float fill)
    : w_(w), h_(h), data_(static_cast<size_t>(w) * h, fill) {}

void Image32f::fill(float v) { std::fill(data_.begin(), data_.end(), v); }

Image16s::Image16s(int w, int h, int16_t fill)
    : w_(w), h_(h), data_(static_cast<size_t>(w) * h, fill) {}

void Image16s::fill(int16_t v) { std::fill(data_.begin(), data_.end(), v); }

Image8u1::Image8u1(int w, int h, uint8_t fill)
    : w_(w), h_(h), data_(static_cast<size_t>(w) * h, fill) {}

void Image8u1::fill(uint8_t v) { std::fill(data_.begin(), data_.end(), v); }

namespace {

bool read_token(std::ifstream& ifs, std::string& tok) {
    for (;;) {
        if (!(ifs >> tok)) return false;
        if (tok[0] == '#') {
            std::string rest;
            std::getline(ifs, rest);
            continue;
        }
        return true;
    }
}

} // namespace

bool load_image(const std::string& path, Image8& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::string magic;
    if (!read_token(ifs, magic)) return false;
    std::string ws, hs, maxs;
    if (!read_token(ifs, ws) || !read_token(ifs, hs) || !read_token(ifs, maxs)) return false;
    const int w = std::stoi(ws);
    const int h = std::stoi(hs);
    const int maxv = std::stoi(maxs);
    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 255) return false;
    ifs.get();
    if (magic == "P5") {
        out = Image8(w, h, 1);
        ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(w) * h);
        return static_cast<bool>(ifs);
    }
    if (magic == "P6") {
        out = Image8(w, h, 3);
        ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(w) * h * 3);
        return static_cast<bool>(ifs);
    }
    return false;
}

bool save_pgm(const std::string& path, const Image8& img) {
    if (img.empty() || img.channels() != 1) return false;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P5\n" << img.width() << " " << img.height() << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()),
              static_cast<std::streamsize>(img.width()) * img.height());
    return static_cast<bool>(ofs);
}

bool save_pgm16(const std::string& path, const Image16s& disp, int scale) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P5\n" << disp.width() << " " << disp.height() << "\n65535\n";
    const int n = disp.width() * disp.height();
    std::vector<uint8_t> raw(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        int v = disp.data()[i];
        if (v < 0) v = 0;
        (void)scale;
        const uint16_t u = static_cast<uint16_t>(clampi(v, 0, 65535));
        raw[static_cast<size_t>(i) * 2] = static_cast<uint8_t>(u >> 8);
        raw[static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>(u & 0xff);
    }
    ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    return static_cast<bool>(ofs);
}

bool save_disparity_preview(const std::string& path, const Image32f& disp, float dmax) {
    if (disp.empty()) return false;
    Image8 preview(disp.width(), disp.height(), 1);
    const float inv = (dmax > 1.f) ? 255.f / dmax : 1.f;
    for (int y = 0; y < disp.height(); ++y) {
        for (int x = 0; x < disp.width(); ++x) {
            const float d = disp.at(x, y);
            if (d < 0.f) preview.at(x, y) = 0;
            else preview.at(x, y) = static_cast<uint8_t>(clampi(static_cast<int>(d * inv + 0.5f), 0, 255));
        }
    }
    return save_pgm(path, preview);
}

bool load_pfm(const std::string& path, Image32f& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    auto read_token = [](std::ifstream& is, std::string& tok) -> bool {
        tok.clear();
        int c = 0;
        while ((c = is.get()) != EOF) {
            if (c == '#') {
                while ((c = is.get()) != EOF && c != '\n');
                continue;
            }
            if (!std::isspace(c)) {
                tok.push_back(static_cast<char>(c));
                break;
            }
        }
        if (tok.empty()) return false;
        while ((c = is.get()) != EOF) {
            if (std::isspace(c)) {
                break;
            }
            tok.push_back(static_cast<char>(c));
        }
        return true;
    };

    std::string tag, sw, sh, sscale;
    if (!read_token(ifs, tag) || !read_token(ifs, sw) || !read_token(ifs, sh) || !read_token(ifs, sscale)) {
        return false;
    }

    if (tag != "Pf" && tag != "PF") {
        return false;
    }
    const int channels = (tag == "PF") ? 3 : 1;
    const int w = std::stoi(sw);
    const int h = std::stoi(sh);
    const float scale = std::stof(sscale);
    if (w <= 0 || h <= 0) return false;

    const bool need_byteswap = (scale > 0.0f);

    out = Image32f(w, h, 0.0f);
    std::vector<float> row_buf(static_cast<size_t>(w) * channels);

    for (int y = h - 1; y >= 0; --y) {
        ifs.read(reinterpret_cast<char*>(row_buf.data()), static_cast<std::streamsize>(row_buf.size() * sizeof(float)));
        if (!ifs) return false;

        if (need_byteswap) {
            for (float& val : row_buf) {
                uint32_t u;
                std::memcpy(&u, &val, sizeof(float));
                u = ((u >> 24) & 0xff) | ((u >> 8) & 0xff00) | ((u << 8) & 0xff0000) | ((u << 24) & 0xff000000);
                std::memcpy(&val, &u, sizeof(float));
            }
        }

        for (int x = 0; x < w; ++x) {
            out.at(x, y) = row_buf[static_cast<size_t>(x) * channels];
        }
    }
    return true;
}

bool save_pfm(const std::string& path, const Image32f& img) {
    if (img.empty()) return false;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    ofs << "Pf\n" << img.width() << " " << img.height() << "\n-1.0\n";
    for (int y = img.height() - 1; y >= 0; --y) {
        ofs.write(reinterpret_cast<const char*>(img.data() + static_cast<size_t>(y) * img.width()),
                  static_cast<std::streamsize>(img.width() * sizeof(float)));
    }
    return static_cast<bool>(ofs);
}

} // namespace apg
