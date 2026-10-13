#include "apg_sgm/c_api.h"
#include "apg_sgm/session.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static void test_c_api_lifecycle() {
    apg_session_t session = apg_session_create(APG_QUALITY_BALANCED, APG_MEMORY_RELEASE_PER_FRAME);
    assert(session != nullptr);
    assert(apg_session_get_frame_count(session) == 0);
    assert(apg_session_get_total_time_ms(session) == 0.0);
    assert(apg_session_get_allocated_bytes(session) == 0);

    apg_result_t result = apg_result_create();
    assert(result != nullptr);

    int w = -1, h = -1;
    assert(apg_result_get_disparity(result, &w, &h) == nullptr);
    assert(w == 0 && h == 0);

    apg_result_destroy(result);
    apg_session_destroy(session);
    std::cout << "[PASS] test_c_api_lifecycle\n";
}

static void test_c_api_execution_and_parity() {
    const int w = 64;
    const int h = 48;
    const int stride = 80; // strided padding

    std::vector<uint8_t> left_mem(stride * h);
    std::vector<uint8_t> right_mem(stride * h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            left_mem[y * stride + x] = static_cast<uint8_t>((x * 3 + y * 7) % 256);
            int shift = (x >= 4) ? 4 : 0;
            right_mem[y * stride + x] = static_cast<uint8_t>(((x - shift) * 3 + y * 7) % 256);
        }
    }

    apg_image_view_t left_v = {left_mem.data(), w, h, stride, 1};
    apg_image_view_t right_v = {right_mem.data(), w, h, stride, 1};

    apg_session_t session = apg_session_create(APG_QUALITY_FAST, APG_MEMORY_RELEASE_PER_FRAME);
    apg_result_t result = apg_result_create();

    int status = apg_session_process(session, &left_v, &right_v, result);
    assert(status == APG_STATUS_OK);
    assert(apg_session_get_frame_count(session) == 1);
    assert(apg_session_get_total_time_ms(session) > 0.0);

    int out_w = 0, out_h = 0;
    const float* disp_c = apg_result_get_disparity(result, &out_w, &out_h);
    assert(disp_c != nullptr);
    assert(out_w == w && out_h == h);

    const float* conf_c = apg_result_get_confidence(result, nullptr, nullptr);
    assert(conf_c != nullptr);

    const uint8_t* inv_c = apg_result_get_invalid_reason(result, nullptr, nullptr);
    assert(inv_c != nullptr);

    const uint8_t* mask_c = apg_result_get_reliable_mask(result, nullptr, nullptr);
    assert(mask_c != nullptr);

    // Parity verification with direct C++ StereoSession
    apg::StereoSession cpp_session(apg::PipelineConfig::from_mode(apg::QualityMode::Fast),
                                   apg::ExecutionOptions{apg::MemoryPolicy::ReleasePerFrame, 0, 0});
    apg::ImageView8 l_cpp(left_mem.data(), w, h, stride, 1);
    apg::ImageView8 r_cpp(right_mem.data(), w, h, stride, 1);
    apg::StereoResult cpp_result;
    bool ok = cpp_session.process(l_cpp, r_cpp, cpp_result);
    assert(ok);

    for (int i = 0; i < w * h; ++i) {
        assert(disp_c[i] == cpp_result.disparity.data()[i]);
        assert(conf_c[i] == cpp_result.confidence.data()[i]);
        assert(inv_c[i] == cpp_result.invalid_reason.data()[i]);
        assert(mask_c[i] == cpp_result.reliable_mask.data()[i]);
    }

    // Reset test
    apg_session_reset(session);
    assert(apg_session_get_frame_count(session) == 0);
    assert(apg_session_get_total_time_ms(session) == 0.0);

    apg_result_destroy(result);
    apg_session_destroy(session);
    std::cout << "[PASS] test_c_api_execution_and_parity\n";
}

static void test_c_api_error_handling() {
    apg_session_t session = apg_session_create(APG_QUALITY_BALANCED, APG_MEMORY_RELEASE_PER_FRAME);
    apg_result_t result = apg_result_create();

    // Null input
    int status = apg_session_process(session, nullptr, nullptr, result);
    assert(status == APG_STATUS_EMPTY_INPUT);
    assert(std::strlen(apg_session_get_last_error(session)) > 0);

    // Size mismatch
    uint8_t dummy[100];
    apg_image_view_t v1 = {dummy, 10, 10, 10, 1};
    apg_image_view_t v2 = {dummy, 20, 10, 20, 1};
    status = apg_session_process(session, &v1, &v2, result);
    assert(status == APG_STATUS_SIZE_MISMATCH);
    assert(std::strcmp(apg_status_string(status), "SizeMismatch") == 0);

    apg_result_destroy(result);
    apg_session_destroy(session);
    std::cout << "[PASS] test_c_api_error_handling\n";
}

static void test_c_api_metadata() {
    assert(std::strcmp(apg_version_string(), "1.0.0") == 0);
    assert(std::strcmp(apg_status_string(APG_STATUS_OK), "Ok") == 0);
    assert(std::strcmp(apg_status_string(APG_STATUS_RESOURCE_LIMIT_EXCEEDED), "ResourceLimitExceeded") == 0);
    std::cout << "[PASS] test_c_api_metadata\n";
}

static void test_c_api_create_ex() {
    apg_session_options_t opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.struct_size = static_cast<uint32_t>(sizeof(opts));
    opts.quality_mode = APG_QUALITY_BALANCED;
    opts.memory_policy = APG_MEMORY_BOUNDED_REUSE;
    opts.max_retained_bytes = 50 * 1024 * 1024;
    opts.num_threads = 4;
    opts.backend = 2; // Packed
    opts.min_disparity = 0;
    opts.max_disparity = 64;

    // Test rejection of struct size mismatch
    opts.struct_size = 1234;
    assert(apg_session_create_ex(&opts) == nullptr);
    opts.struct_size = static_cast<uint32_t>(sizeof(opts));

    apg_session_t session = apg_session_create_ex(&opts);
    assert(session != nullptr);

    apg_session_destroy(session);
    std::cout << "[PASS] test_c_api_create_ex\n";
}

int main() {
    test_c_api_lifecycle();
    test_c_api_execution_and_parity();
    test_c_api_error_handling();
    test_c_api_metadata();
    test_c_api_create_ex();
    std::cout << "All C API tests passed successfully!\n";
    return 0;
}
