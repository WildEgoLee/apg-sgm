#pragma once

#include "apg_sgm/types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace apg {

enum class InpaintMethod {
    Legacy,          // Production baseline (100% bit-exact, fast-path)
    ShadowReclass    // Winning P5.6 candidate: Shadow-aware mismatch fill with depth step threshold
};

inline std::string inpaint_method_to_string(InpaintMethod m) {
    switch (m) {
        case InpaintMethod::Legacy: return "legacy";
        case InpaintMethod::ShadowReclass: return "shadow_reclass";
    }
    return "unknown";
}

inline bool parse_inpaint_method(const std::string& str, InpaintMethod& out) {
    if (str == "legacy" || str == "A" || str == "a") {
        out = InpaintMethod::Legacy;
        return true;
    }
    if (str == "shadow_reclass" || str == "shadow" || str == "C" || str == "c") {
        out = InpaintMethod::ShadowReclass;
        return true;
    }
    return false;
}

enum class FillRoute : uint8_t {
    None = 0,
    OccBothMin = 1,
    OccLeftOnly = 2,
    OccRightOnly = 3,
    MismatchColorHoriz = 4,
    MismatchColorVert = 5,
    MismatchPrior = 6,
    MismatchLerp = 7,
    MismatchLeftOnly = 8,
    MismatchRightOnly = 9,
    RemainedInvalid = 10,
    Count = 11
};

inline const char* fill_route_to_string(FillRoute r) {
    switch (r) {
        case FillRoute::None: return "none";
        case FillRoute::OccBothMin: return "occ_both_min";
        case FillRoute::OccLeftOnly: return "occ_left_only";
        case FillRoute::OccRightOnly: return "occ_right_only";
        case FillRoute::MismatchColorHoriz: return "mismatch_color_horiz";
        case FillRoute::MismatchColorVert: return "mismatch_color_vert";
        case FillRoute::MismatchPrior: return "mismatch_prior";
        case FillRoute::MismatchLerp: return "mismatch_lerp";
        case FillRoute::MismatchLeftOnly: return "mismatch_left_only";
        case FillRoute::MismatchRightOnly: return "mismatch_right_only";
        case FillRoute::RemainedInvalid: return "remained_invalid";
        default: return "unknown";
    }
}

struct InpaintTelemetry {
    uint64_t total_invalid_pixels = 0;
    uint64_t classified_occ_pixels = 0;
    uint64_t classified_mismatch_pixels = 0;
    uint64_t hit_max_gap_pixels = 0;
    uint64_t reclassified_shadow_pixels = 0;
    uint64_t route_counts[11] = {0};

    // Run length statistics
    uint64_t invalid_runs_count = 0;
    uint64_t invalid_runs_truncated = 0;
    uint64_t run_length_buckets[10] = {0}; // [0,4), [4,8), [8,16), [16,24), [24,32), [32,48), [48,64), [64,96), [96,128), [128, inf)

    void reset() {
        total_invalid_pixels = 0;
        classified_occ_pixels = 0;
        classified_mismatch_pixels = 0;
        hit_max_gap_pixels = 0;
        reclassified_shadow_pixels = 0;
        for (int i = 0; i < 11; ++i) route_counts[i] = 0;
        invalid_runs_count = 0;
        invalid_runs_truncated = 0;
        for (int i = 0; i < 10; ++i) run_length_buckets[i] = 0;
    }

    void record_run(int len, int max_gap) {
        invalid_runs_count++;
        if (len > max_gap) {
            invalid_runs_truncated++;
        }
        if (len < 4) run_length_buckets[0]++;
        else if (len < 8) run_length_buckets[1]++;
        else if (len < 16) run_length_buckets[2]++;
        else if (len < 24) run_length_buckets[3]++;
        else if (len < 32) run_length_buckets[4]++;
        else if (len < 48) run_length_buckets[5]++;
        else if (len < 64) run_length_buckets[6]++;
        else if (len < 96) run_length_buckets[7]++;
        else if (len < 128) run_length_buckets[8]++;
        else run_length_buckets[9]++;
    }

    void merge(const InpaintTelemetry& other) {
        total_invalid_pixels += other.total_invalid_pixels;
        classified_occ_pixels += other.classified_occ_pixels;
        classified_mismatch_pixels += other.classified_mismatch_pixels;
        hit_max_gap_pixels += other.hit_max_gap_pixels;
        reclassified_shadow_pixels += other.reclassified_shadow_pixels;
        for (int i = 0; i < 11; ++i) {
            route_counts[i] += other.route_counts[i];
        }
        invalid_runs_count += other.invalid_runs_count;
        invalid_runs_truncated += other.invalid_runs_truncated;
        for (int i = 0; i < 10; ++i) {
            run_length_buckets[i] += other.run_length_buckets[i];
        }
    }

    uint64_t total_filled_pixels() const {
        const uint64_t rem = route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)];
        return (total_invalid_pixels > rem) ? (total_invalid_pixels - rem) : 0;
    }

    double fill_ratio() const {
        return total_invalid_pixels > 0 ? static_cast<double>(total_filled_pixels()) / total_invalid_pixels : 0.0;
    }

    uint64_t occ_filled_pixels() const {
        return route_counts[static_cast<size_t>(FillRoute::OccBothMin)]
             + route_counts[static_cast<size_t>(FillRoute::OccLeftOnly)]
             + route_counts[static_cast<size_t>(FillRoute::OccRightOnly)];
    }

    uint64_t occ_unfilled_pixels() const {
        // Native occ pixels that remained unfilled
        return (classified_occ_pixels + reclassified_shadow_pixels > occ_filled_pixels())
             ? ((classified_occ_pixels + reclassified_shadow_pixels) - occ_filled_pixels()) : 0;
    }

    uint64_t mismatch_native_filled_pixels() const {
        return route_counts[static_cast<size_t>(FillRoute::MismatchColorHoriz)]
             + route_counts[static_cast<size_t>(FillRoute::MismatchColorVert)]
             + route_counts[static_cast<size_t>(FillRoute::MismatchPrior)]
             + route_counts[static_cast<size_t>(FillRoute::MismatchLerp)]
             + route_counts[static_cast<size_t>(FillRoute::MismatchLeftOnly)]
             + route_counts[static_cast<size_t>(FillRoute::MismatchRightOnly)];
    }

    uint64_t mismatch_filled_pixels() const {
        // Includes shadow-reclassified pixels, which were originally classified as Mismatch
        return mismatch_native_filled_pixels() + reclassified_shadow_pixels;
    }

    uint64_t mismatch_unfilled_pixels() const {
        // Mismatch holes that remained unfilled: must exclude reclassified shadow pixels that were successfully filled
        const uint64_t filled = mismatch_filled_pixels();
        return (classified_mismatch_pixels > filled) ? (classified_mismatch_pixels - filled) : 0;
    }
};

class ExperimentalInpaintScope {
public:
    explicit ExperimentalInpaintScope(
        InpaintMethod method = InpaintMethod::Legacy,
        int max_gap = 32,
        bool collect_telemetry = false
    );

    ~ExperimentalInpaintScope();

    static InpaintMethod current_method();
    static int current_max_gap();
    static bool is_telemetry_enabled();
    static InpaintTelemetry& current_telemetry();

private:
    InpaintMethod prev_method_;
    int prev_max_gap_;
    bool prev_telem_enabled_;
};

} // namespace apg
