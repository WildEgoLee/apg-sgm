#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace apg {

enum class SubpixelMethod {
    Legacy,       // A: Legacy parabolic/quadratic interpolation (production baseline)
    CurvGate,     // B: Curvature-gated quadratic
    ConfGate,     // C: Confidence/uniqueness-gated quadratic
    Stabilized    // D: Stabilized quadratic with denominator regularization
};

inline std::string subpixel_method_to_string(SubpixelMethod m) {
    switch (m) {
        case SubpixelMethod::Legacy: return "legacy";
        case SubpixelMethod::CurvGate: return "curv_gate";
        case SubpixelMethod::ConfGate: return "conf_gate";
        case SubpixelMethod::Stabilized: return "stabilized";
    }
    return "unknown";
}

inline bool parse_subpixel_method(const std::string& str, SubpixelMethod& out) {
    if (str == "legacy" || str == "A" || str == "a") {
        out = SubpixelMethod::Legacy;
        return true;
    }
    if (str == "curv_gate" || str == "curv" || str == "B" || str == "b") {
        out = SubpixelMethod::CurvGate;
        return true;
    }
    if (str == "conf_gate" || str == "conf" || str == "C" || str == "c") {
        out = SubpixelMethod::ConfGate;
        return true;
    }
    if (str == "stabilized" || str == "stab" || str == "D" || str == "d") {
        out = SubpixelMethod::Stabilized;
        return true;
    }
    return false;
}

struct SubpixelTelemetry {
    uint64_t candidate_pixels = 0;
    uint64_t applied_pixels = 0;
    uint64_t degenerate_count = 0;
    uint64_t clamped_count = 0;
    uint64_t negative_curvature_count = 0;
    double sum_abs_delta = 0.0;
    uint64_t delta_buckets[50] = {0}; // 50 buckets of width 0.01 in [0.0, 0.5]

    void reset() {
        candidate_pixels = 0;
        applied_pixels = 0;
        degenerate_count = 0;
        clamped_count = 0;
        negative_curvature_count = 0;
        sum_abs_delta = 0.0;
        for (int i = 0; i < 50; ++i) delta_buckets[i] = 0;
    }

    void merge(const SubpixelTelemetry& other) {
        candidate_pixels += other.candidate_pixels;
        applied_pixels += other.applied_pixels;
        degenerate_count += other.degenerate_count;
        clamped_count += other.clamped_count;
        negative_curvature_count += other.negative_curvature_count;
        sum_abs_delta += other.sum_abs_delta;
        for (int i = 0; i < 50; ++i) {
            delta_buckets[i] += other.delta_buckets[i];
        }
    }

    double applied_ratio() const {
        return candidate_pixels > 0 ? static_cast<double>(applied_pixels) / candidate_pixels : 0.0;
    }

    double clamped_ratio() const {
        return applied_pixels > 0 ? static_cast<double>(clamped_count) / applied_pixels : 0.0;
    }

    double mean_abs_delta() const {
        return applied_pixels > 0 ? sum_abs_delta / applied_pixels : 0.0;
    }

    float p50_abs_delta() const {
        return percentile_abs_delta(0.50f);
    }

    float p95_abs_delta() const {
        return percentile_abs_delta(0.95f);
    }

    float percentile_abs_delta(float q) const {
        if (applied_pixels == 0) return 0.0f;
        uint64_t target = static_cast<uint64_t>(std::ceil(q * applied_pixels));
        uint64_t acc = 0;
        for (int i = 0; i < 50; ++i) {
            acc += delta_buckets[i];
            if (acc >= target) {
                return (i + 0.5f) * 0.01f;
            }
        }
        return 0.5f;
    }
};

class SubpixelEstimator {
public:
    static inline float compute_legacy_delta(float cm, float c0, float cp) noexcept {
        const float denom = cm - 2.0f * c0 + cp;
        if (std::abs(denom) > 1e-6f) {
            float delta = 0.5f * (cm - cp) / denom;
            if (delta < -0.5f) return -0.5f;
            if (delta > 0.5f) return 0.5f;
            return delta;
        }
        return 0.0f;
    }

    static float compute_delta(
        float cm, float c0, float cp,
        SubpixelMethod method = SubpixelMethod::Legacy,
        float param = 0.0f,
        float uniqueness = 1.0f,
        SubpixelTelemetry* telem = nullptr
    );
};

class ExperimentalSubpixelScope {
public:
    explicit ExperimentalSubpixelScope(
        SubpixelMethod method = SubpixelMethod::Legacy,
        float param = 0.0f,
        bool collect_telemetry = false
    );

    ~ExperimentalSubpixelScope();

    static SubpixelMethod current_method();
    static float current_param();
    static bool is_telemetry_enabled();
    static SubpixelTelemetry& current_telemetry();

private:
    SubpixelMethod prev_method_;
    float prev_param_;
    bool prev_telem_enabled_;
};

} // namespace apg
