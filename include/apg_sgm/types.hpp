#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace apg {

enum class QualityMode {
    Fast,
    Balanced,
    HighQuality
};

enum class CensusType {
    Census9x7,
    SymmetricCensus9x7
};

enum class PathType {
    Path4,
    Path8
};

enum class InvalidReason : uint8_t {
    Valid = 0,
    Occlusion,
    Mismatch,
    LowConfidence,
    OutOfRange
};

constexpr int kSubpixelShift = 4;
constexpr int kSubpixelScale = 1 << kSubpixelShift;
constexpr int16_t kInvalidDisp = -1;
constexpr uint16_t kInvalidCost = 65535;
constexpr uint32_t kInvalidCost32 = 0xFFFFFFFFu;
constexpr int kPathInf = 1 << 28;

template <typename T>
constexpr T invalid_cost();

template <>
constexpr uint16_t invalid_cost<uint16_t>() {
    return kInvalidCost;
}

template <>
constexpr uint32_t invalid_cost<uint32_t>() {
    return kInvalidCost32;
}

struct Size {
    int width = 0;
    int height = 0;
    int pixels() const { return width * height; }
    bool empty() const { return width <= 0 || height <= 0; }
};

struct Pixel {
    int x = 0;
    int y = 0;
};

struct SupportMatch {
    int x = 0;
    int y = 0;
    float disparity = 0.f;
    float confidence = 0.f;
};

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace apg
