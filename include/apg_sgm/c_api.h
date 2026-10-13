#ifndef APG_SGM_C_API_H
#define APG_SGM_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(APG_SGM_BUILDING_DLL)
    #define APG_API __declspec(dllexport)
  #elif defined(APG_SGM_USING_DLL)
    #define APG_API __declspec(dllimport)
  #else
    #define APG_API
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define APG_API __attribute__((visibility("default")))
  #else
    #define APG_API
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Quality Mode */
#define APG_QUALITY_FAST 0
#define APG_QUALITY_BALANCED 1
#define APG_QUALITY_HIGH_QUALITY 2

/* Memory Policy */
#define APG_MEMORY_RELEASE_PER_FRAME 0
#define APG_MEMORY_BOUNDED_REUSE 1

/* Status Codes */
typedef enum {
    APG_STATUS_OK = 0,
    APG_STATUS_EMPTY_INPUT = 1,
    APG_STATUS_SIZE_MISMATCH = 2,
    APG_STATUS_INVALID_DISPARITY_RANGE = 3,
    APG_STATUS_INVALID_CONFIG = 4,
    APG_STATUS_UNSUPPORTED_BACKEND = 5,
    APG_STATUS_RESOURCE_LIMIT_EXCEEDED = 6,
    APG_STATUS_ALLOCATION_FAILURE = 7,
    APG_STATUS_INTERNAL_ERROR = 8
} apg_status_code_t;

/* Strided Raw Image Descriptor */
typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int stride;
    int channels;
} apg_image_view_t;

/* Opaque handles */
typedef struct apg_session_s* apg_session_t;
typedef struct apg_result_s* apg_result_t;

/* Lifecycle */
APG_API apg_session_t apg_session_create(int quality_mode, int memory_policy);
APG_API void apg_session_destroy(apg_session_t session);
APG_API void apg_session_reset(apg_session_t session);

APG_API apg_result_t apg_result_create(void);
APG_API void apg_result_destroy(apg_result_t result);

/* Execution */
APG_API int apg_session_process(apg_session_t session,
                                const apg_image_view_t* left,
                                const apg_image_view_t* right,
                                apg_result_t result);

/* Result access */
APG_API const float* apg_result_get_disparity(apg_result_t result, int* width, int* height);
APG_API const float* apg_result_get_confidence(apg_result_t result, int* width, int* height);
APG_API const uint8_t* apg_result_get_invalid_reason(apg_result_t result, int* width, int* height);
APG_API const uint8_t* apg_result_get_reliable_mask(apg_result_t result, int* width, int* height);

/* Diagnostics & Metadata */
APG_API const char* apg_session_get_last_error(apg_session_t session);
APG_API size_t apg_session_get_allocated_bytes(apg_session_t session);
APG_API size_t apg_session_get_frame_count(apg_session_t session);
APG_API double apg_session_get_total_time_ms(apg_session_t session);

APG_API const char* apg_status_string(int status_code);
APG_API const char* apg_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* APG_SGM_C_API_H */
