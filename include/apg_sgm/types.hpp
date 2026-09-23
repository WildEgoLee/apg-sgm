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
