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

    apg_session_s(apg::PipelineConfig cfg, apg::ExecutionOptions opt)
        : session(std::move(cfg), opt) {}
};

struct apg_result_s {
    apg::StereoResult result;
};

apg_session_t apg_session_create(int quality_mode, int memory_policy) {
    apg_session_options_t opts = {};
    opts.quality_mode = quality_mode;
    opts.memory_policy = memory_policy;
    opts.max_retained_bytes = 0;
    opts.num_threads = 0;
    opts.backend = 0;
    opts.min_disparity = 0;
    opts.max_disparity = 128;
    return apg_session_create_ex(&opts);
}

apg_session_t apg_session_create_ex(const apg_session_options_t* options) {
    if (!options) return nullptr;
    try {
        apg::QualityMode qm = apg::QualityMode::Balanced;
        if (options->quality_mode == APG_QUALITY_FAST) {
            qm = apg::QualityMode::Fast;
        } else if (options->quality_mode == APG_QUALITY_HIGH_QUALITY) {
            qm = apg::QualityMode::HighQuality;
        }

        int max_disp = options->max_disparity > options->min_disparity ? options->max_disparity : 128;
        auto cfg = apg::PipelineConfig::from_mode(qm, max_disp);
        cfg.min_disparity = options->min_disparity;
        cfg.max_disparity = max_disp;
        if (options->backend == 1) {
            cfg.volume_backend = apg::VolumeBackend::Dense;
        } else if (options->backend == 2) {
            cfg.volume_backend = apg::VolumeBackend::Packed;
        } else {
            cfg.volume_backend = apg::VolumeBackend::Auto;
        }
        cfg.num_threads = options->num_threads;

        apg::MemoryPolicy mp = (options->memory_policy == APG_MEMORY_BOUNDED_REUSE)
                                   ? apg::MemoryPolicy::BoundedReuse
                                   : apg::MemoryPolicy::ReleasePerFrame;

        apg::ExecutionOptions exec_opts;
        exec_opts.memory_policy = mp;
        exec_opts.max_retained_bytes = options->max_retained_bytes;
        exec_opts.num_threads = options->num_threads;

        return new apg_session_s(std::move(cfg), exec_opts);
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
