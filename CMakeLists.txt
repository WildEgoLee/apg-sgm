cmake_minimum_required(VERSION 3.16)
project(apg_sgm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(apg_sgm
    src/image.cpp
    src/config.cpp
    src/cost_computer.cpp
    src/cost_computer_avx2.cpp
    src/prior_estimator.cpp
    src/cost_aggregator.cpp
    src/cost_aggregator_avx2.cpp
    src/sgm_optimizer.cpp
    src/sgm_optimizer_avx2.cpp
    src/confidence_estimator.cpp
    src/refiner.cpp
    src/refiner_avx2.cpp
    src/postprocess.cpp
    src/pipeline.cpp
    src/ablation.cpp
    src/metrics.cpp
)
target_include_directories(apg_sgm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(apg_sgm PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(apg_sgm PUBLIC APG_SGM_OPENMP=1)
endif()

if(MSVC)
    target_compile_options(apg_sgm PRIVATE /W4)
    set_source_files_properties(src/sgm_optimizer_avx2.cpp src/cost_aggregator_avx2.cpp src/cost_computer_avx2.cpp src/refiner_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    target_compile_options(apg_sgm PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(src/sgm_optimizer_avx2.cpp src/cost_aggregator_avx2.cpp src/cost_computer_avx2.cpp src/refiner_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

add_executable(apg-stereo apps/run_stereo.cpp)
target_link_libraries(apg-stereo PRIVATE apg_sgm)

add_executable(apg-benchmark apps/benchmark_stereo.cpp)
target_link_libraries(apg-benchmark PRIVATE apg_sgm)

enable_testing()
add_executable(apg_sgm_tests tests/test_sanity.cpp)
target_include_directories(apg_sgm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(apg_sgm_tests PRIVATE apg_sgm)
add_test(NAME sanity COMMAND apg_sgm_tests)

add_executable(test_synthetic tests/test_synthetic.cpp)
target_link_libraries(test_synthetic PRIVATE apg_sgm)
add_test(NAME synthetic COMMAND test_synthetic)

add_executable(test_ablation tests/test_ablation.cpp)
target_link_libraries(test_ablation PRIVATE apg_sgm)
add_test(NAME ablation COMMAND test_ablation)

add_executable(test_headers tests/test_headers.cpp)
target_link_libraries(test_headers PRIVATE apg_sgm)
add_test(NAME headers COMMAND test_headers)

add_executable(test_api_contract tests/test_api_contract.cpp)
target_link_libraries(test_api_contract PRIVATE apg_sgm)
add_test(NAME api_contract COMMAND test_api_contract)

add_executable(test_status tests/test_status.cpp)
target_link_libraries(test_status PRIVATE apg_sgm)
add_test(NAME status COMMAND test_status)

add_executable(test_workspace tests/test_workspace.cpp)
target_link_libraries(test_workspace PRIVATE apg_sgm)
add_test(NAME workspace COMMAND test_workspace)

add_executable(test_session tests/test_session.cpp)
target_link_libraries(test_session PRIVATE apg_sgm)
add_test(NAME session COMMAND test_session)

add_executable(test_image_view tests/test_image_view.cpp)
target_link_libraries(test_image_view PRIVATE apg_sgm)
add_test(NAME image_view COMMAND test_image_view)

add_executable(test_observability tests/test_observability.cpp)
target_link_libraries(test_observability PRIVATE apg_sgm)
add_test(NAME observability COMMAND test_observability)
