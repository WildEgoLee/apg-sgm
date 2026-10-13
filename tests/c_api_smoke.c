#include "apg_sgm/c_api.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("Starting Pure C API Smoke Test...\n");

    /* Version check */
    const char* ver = apg_version_string();
    assert(ver != NULL);
    assert(strcmp(ver, "1.0.0") == 0);

    /* Session creation via extended options */
    apg_session_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.quality_mode = APG_QUALITY_FAST;
    opts.memory_policy = APG_MEMORY_BOUNDED_REUSE;
    opts.max_retained_bytes = 100 * 1024 * 1024; /* 100 MB budget */
    opts.num_threads = 2;
    opts.backend = 0; /* Auto */
    opts.min_disparity = 0;
    opts.max_disparity = 32;

    apg_session_t session = apg_session_create_ex(&opts);
    assert(session != NULL);

    /* Synthetic small strided images */
    const int w = 32;
    const int h = 24;
    const int stride = 40;
    uint8_t* left_data = (uint8_t*)malloc(stride * h);
    uint8_t* right_data = (uint8_t*)malloc(stride * h);
    assert(left_data != NULL && right_data != NULL);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            left_data[y * stride + x] = (uint8_t)((x * 5 + y * 11) % 256);
            int d = (x >= 2) ? 2 : 0;
            right_data[y * stride + x] = (uint8_t)(((x - d) * 5 + y * 11) % 256);
        }
    }

    apg_image_view_t left_v;
    left_v.data = left_data;
    left_v.width = w;
    left_v.height = h;
    left_v.stride = stride;
    left_v.channels = 1;

    apg_image_view_t right_v;
    right_v.data = right_data;
    right_v.width = w;
    right_v.height = h;
    right_v.stride = stride;
    right_v.channels = 1;

    apg_result_t result = apg_result_create();
    assert(result != NULL);

    /* Process frame 1 */
    int rc = apg_session_process(session, &left_v, &right_v, result);
    assert(rc == APG_STATUS_OK);
    assert(apg_session_get_frame_count(session) == 1);

    int out_w = 0, out_h = 0;
    const float* disp = apg_result_get_disparity(result, &out_w, &out_h);
    assert(disp != NULL);
    assert(out_w == w && out_h == h);

    const float* conf = apg_result_get_confidence(result, NULL, NULL);
    assert(conf != NULL);

    const uint8_t* inv = apg_result_get_invalid_reason(result, NULL, NULL);
    assert(inv != NULL);

    const uint8_t* mask = apg_result_get_reliable_mask(result, NULL, NULL);
    assert(mask != NULL);

    /* Process frame 2 (test BoundedReuse steady state in pure C) */
    rc = apg_session_process(session, &left_v, &right_v, result);
    assert(rc == APG_STATUS_OK);
    assert(apg_session_get_frame_count(session) == 2);

    /* Teardown */
    apg_result_destroy(result);
    apg_session_destroy(session);
    free(left_data);
    free(right_data);

    printf("Pure C API smoke test passed successfully!\n");
    return 0;
}
