#include "apg_sgm/c_api.h"
#include "apg_sgm/image_view.hpp"
#include "apg_sgm/session.hpp"
#include "apg_sgm/status.hpp"

#include <exception>
#include <new>
#include <string>

struct apg_session_s {
    apg::StereoSession session;
    std::string last_error;

    apg_session_s(apg::QualityMode qm, apg::MemoryPolicy mp)
        : session(apg::PipelineConfig::from_mode(qm), apg::ExecutionOptions{mp, 0, 0}) {}
};

struct apg_result_s {
    apg::StereoResult result;
};

apg_session_t apg_session_create(int quality_mode, int memory_policy) {
    try {
        apg::QualityMode qm = apg::QualityMode::Balanced;
        if (quality_mode == APG_QUALITY_FAST) {
            qm = apg::QualityMode::Fast;
        } else if (quality_mode == APG_QUALITY_HIGH_QUALITY) {
            qm = apg::QualityMode::HighQuality;
        }

        apg::MemoryPolicy mp = (memory_policy == APG_MEMORY_BOUNDED_REUSE)
                                   ? apg::MemoryPolicy::BoundedReuse
                                   : apg::MemoryPolicy::ReleasePerFrame;

        return new apg_session_s(qm, mp);
    } catch (...) {
        return nullptr;
    }
}

void apg_session_destroy(apg_session_t session) {
    delete session;
}

void apg_session_reset(apg_session_t session) {
    if (session) {
        session->session.reset();
        session->last_error.clear();
    }
}

apg_result_t apg_result_create(void) {
    try {
        return new apg_result_s();
    } catch (...) {
        return nullptr;
    }
}

void apg_result_destroy(apg_result_t result) {
    delete result;
}

int apg_session_process(apg_session_t session,
                        const apg_image_view_t* left,
                        const apg_image_view_t* right,
                        apg_result_t result) {
    if (!session || !result) {
        return APG_STATUS_INTERNAL_ERROR;
    }
    if (!left || !right || !left->data || !right->data) {
        session->last_error = "Null input pointer";
        return APG_STATUS_EMPTY_INPUT;
    }

    try {
        apg::ImageView8 l_view(left->data, left->width, left->height, left->stride, left->channels);
        apg::ImageView8 r_view(right->data, right->width, right->height, right->stride, right->channels);

        apg::Status s = session->session.process_status(l_view, r_view, result->result);
        if (!s.ok()) {
            session->last_error = s.message;
            return static_cast<int>(s.code);
        }
        session->last_error.clear();
        return APG_STATUS_OK;
    } catch (const std::bad_alloc&) {
        session->last_error = "Memory allocation failure";
        return APG_STATUS_ALLOCATION_FAILURE;
    } catch (const std::exception& e) {
        session->last_error = e.what();
        return APG_STATUS_INTERNAL_ERROR;
    } catch (...) {
        session->last_error = "Unknown error";
        return APG_STATUS_INTERNAL_ERROR;
    }
}

const float* apg_result_get_disparity(apg_result_t result, int* width, int* height) {
    if (!result || result->result.disparity.empty()) {
        if (width) *width = 0;
        if (height) *height = 0;
        return nullptr;
    }
    if (width) *width = result->result.width();
    if (height) *height = result->result.height();
    return result->result.disparity.data();
}

const float* apg_result_get_confidence(apg_result_t result, int* width, int* height) {
    if (!result || result->result.confidence.empty()) {
        if (width) *width = 0;
        if (height) *height = 0;
        return nullptr;
    }
    if (width) *width = result->result.width();
    if (height) *height = result->result.height();
    return result->result.confidence.data();
}

const uint8_t* apg_result_get_invalid_reason(apg_result_t result, int* width, int* height) {
    if (!result || result->result.invalid_reason.empty()) {
        if (width) *width = 0;
        if (height) *height = 0;
        return nullptr;
    }
    if (width) *width = result->result.width();
    if (height) *height = result->result.height();
    return result->result.invalid_reason.data();
}

const uint8_t* apg_result_get_reliable_mask(apg_result_t result, int* width, int* height) {
    if (!result || result->result.reliable_mask.empty()) {
        if (width) *width = 0;
        if (height) *height = 0;
        return nullptr;
    }
    if (width) *width = result->result.width();
    if (height) *height = result->result.height();
    return result->result.reliable_mask.data();
}

const char* apg_session_get_last_error(apg_session_t session) {
    if (!session) return "Invalid session handle";
    return session->last_error.c_str();
}

size_t apg_session_get_allocated_bytes(apg_session_t session) {
    if (!session) return 0;
    return session->session.allocated_bytes();
}

size_t apg_session_get_frame_count(apg_session_t session) {
    if (!session) return 0;
    return session->session.frame_count();
}

double apg_session_get_total_time_ms(apg_session_t session) {
    if (!session) return 0.0;
    return session->session.total_time_ms();
}

const char* apg_status_string(int status_code) {
    return apg::status_code_string(static_cast<apg::StatusCode>(status_code));
}

const char* apg_version_string(void) {
    return "1.0.0";
}
