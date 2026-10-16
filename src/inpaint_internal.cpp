#include "inpaint_internal.hpp"

namespace apg {

namespace {

struct ScopeState {
    InpaintMethod method = InpaintMethod::Legacy;
    int max_gap = 32;
    bool telemetry_enabled = false;
    InpaintTelemetry telemetry;
};

thread_local ScopeState s_inpaint_state;

} // namespace

InpaintMethod ExperimentalInpaintScope::current_method() {
    return s_inpaint_state.method;
}

int ExperimentalInpaintScope::current_max_gap() {
    return s_inpaint_state.max_gap;
}

bool ExperimentalInpaintScope::is_telemetry_enabled() {
    return s_inpaint_state.telemetry_enabled;
}

InpaintTelemetry& ExperimentalInpaintScope::current_telemetry() {
    return s_inpaint_state.telemetry;
}

ExperimentalInpaintScope::ExperimentalInpaintScope(
    InpaintMethod method,
    int max_gap,
    bool collect_telemetry
) {
    prev_method_ = s_inpaint_state.method;
    prev_max_gap_ = s_inpaint_state.max_gap;
    prev_telem_enabled_ = s_inpaint_state.telemetry_enabled;

    s_inpaint_state.method = method;
    s_inpaint_state.max_gap = (max_gap > 0) ? max_gap : 32;
    s_inpaint_state.telemetry_enabled = collect_telemetry;
    if (collect_telemetry) {
        s_inpaint_state.telemetry.reset();
    }
}

ExperimentalInpaintScope::~ExperimentalInpaintScope() {
    s_inpaint_state.method = prev_method_;
    s_inpaint_state.max_gap = prev_max_gap_;
    s_inpaint_state.telemetry_enabled = prev_telem_enabled_;
}

} // namespace apg
