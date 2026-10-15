#include "subpixel_estimator.hpp"

namespace apg {

namespace {

struct ScopeState {
    SubpixelMethod method = SubpixelMethod::Legacy;
    float param = 0.0f;
    bool telemetry_enabled = false;
    SubpixelTelemetry telemetry;
};

thread_local ScopeState s_scope_state;

} // namespace

SubpixelMethod ExperimentalSubpixelScope::current_method() {
    return s_scope_state.method;
}

float ExperimentalSubpixelScope::current_param() {
    return s_scope_state.param;
}

bool ExperimentalSubpixelScope::is_telemetry_enabled() {
    return s_scope_state.telemetry_enabled;
}

SubpixelTelemetry& ExperimentalSubpixelScope::current_telemetry() {
    return s_scope_state.telemetry;
}

ExperimentalSubpixelScope::ExperimentalSubpixelScope(
    SubpixelMethod method,
    float param,
    bool collect_telemetry
) {
    prev_method_ = s_scope_state.method;
    prev_param_ = s_scope_state.param;
    prev_telem_enabled_ = s_scope_state.telemetry_enabled;

    s_scope_state.method = method;
    s_scope_state.param = param;
    s_scope_state.telemetry_enabled = collect_telemetry;
    if (collect_telemetry) {
        s_scope_state.telemetry.reset();
    }
}

ExperimentalSubpixelScope::~ExperimentalSubpixelScope() {
    s_scope_state.method = prev_method_;
    s_scope_state.param = prev_param_;
    s_scope_state.telemetry_enabled = prev_telem_enabled_;
}

float SubpixelEstimator::compute_delta(
    float cm, float c0, float cp,
    SubpixelMethod method,
    float param,
    float uniqueness,
    SubpixelTelemetry* telem
) {
    if (!telem && method == SubpixelMethod::Legacy) {
        return compute_legacy_delta(cm, c0, cp);
    }

    if (telem) {
        telem->candidate_pixels++;
    }

    const float denom = cm - 2.0f * c0 + cp;
    const bool non_convex = (c0 >= cm || c0 >= cp || denom <= 1e-6f);

    if (non_convex && telem) {
        telem->negative_curvature_count++;
        if (std::abs(denom) <= 1e-6f || (cm == c0 && c0 == cp)) {
            telem->degenerate_count++;
        }
    }

    float delta = 0.0f;
    float raw_delta = 0.0f;

    switch (method) {
        case SubpixelMethod::Legacy: {
            if (std::abs(denom) > 1e-6f) {
                raw_delta = 0.5f * (cm - cp) / denom;
                delta = std::clamp(raw_delta, -0.5f, 0.5f);
            }
            break;
        }

        case SubpixelMethod::CurvGate: {
            const float min_curv = (param > 0.0f) ? param : 4.0f;
            if (!non_convex && denom >= min_curv) {
                raw_delta = 0.5f * (cm - cp) / denom;
                delta = std::clamp(raw_delta, -0.5f, 0.5f);
            }
            break;
        }

        case SubpixelMethod::ConfGate: {
            const float min_conf = (param > 0.0f) ? param : 0.05f;
            if (!non_convex && uniqueness >= min_conf) {
                raw_delta = 0.5f * (cm - cp) / denom;
                delta = std::clamp(raw_delta, -0.5f, 0.5f);
            }
            break;
        }

        case SubpixelMethod::Stabilized: {
            const float lambda = (param > 0.0f) ? param : 4.0f;
            if (c0 <= cm && c0 <= cp && denom > 0.0f) {
                raw_delta = 0.5f * (cm - cp) / (denom + lambda);
                delta = std::clamp(raw_delta, -0.5f, 0.5f);
            }
            break;
        }
    }

    if (telem) {
        if (std::abs(delta) > 1e-5f) {
            telem->applied_pixels++;
            const float a = std::abs(delta);
            telem->sum_abs_delta += a;
            int bi = std::min(49, static_cast<int>(a * 100.0f));
            telem->delta_buckets[bi]++;
        }
        if (std::abs(raw_delta) >= 0.5f) {
            telem->clamped_count++;
        }
    }

    return delta;
}

} // namespace apg
