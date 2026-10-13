cmake_minimum_required(VERSION 3.16)
project(apg_sgm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(APG_SGM_BUILD_APPS "Build CLI applications" ON)
option(APG_SGM_BUILD_TESTS "Build test suite" ON)
option(APG_SGM_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(APG_SGM_BUILD_PYTHON "Build Python bindings" OFF)

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
    src/c_api.cpp
)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

target_include_directories(apg_sgm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(apg_sgm PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(apg_sgm PUBLIC APG_SGM_OPENMP=1)
endif()

target_compile_definitions(apg_sgm PRIVATE APG_SGM_BUILDING_DLL)

if(MSVC)
    target_compile_options(apg_sgm PRIVATE /W4)
    set_source_files_properties(src/sgm_optimizer_avx2.cpp src/cost_aggregator_avx2.cpp src/cost_computer_avx2.cpp src/refiner_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    target_compile_options(apg_sgm PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(src/sgm_optimizer_avx2.cpp src/cost_aggregator_avx2.cpp src/cost_computer_avx2.cpp src/refiner_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

if(APG_SGM_BUILD_PYTHON)
    if(NOT BUILD_SHARED_LIBS)
        set_target_properties(apg_sgm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_res
        )
        if(_pybind11_res EQUAL 0)
            find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_dir}")
        else()
            message(FATAL_ERROR "pybind11 could not be located via Python3")
        endif()
    endif()

    pybind11_add_module(_core python/src/binding.cpp)
    target_link_libraries(_core PRIVATE apg_sgm)
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/apg_sgm"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/apg_sgm"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/python/apg_sgm"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/python/apg_sgm"
    )
    install(TARGETS _core DESTINATION apg_sgm)
endif()

if(APG_SGM_BUILD_APPS)
    add_executable(apg-stereo apps/run_stereo.cpp)
    target_link_libraries(apg-stereo PRIVATE apg_sgm)
endif()

if(APG_SGM_BUILD_BENCHMARKS)
    add_executable(apg-benchmark apps/benchmark_stereo.cpp)
    target_link_libraries(apg-benchmark PRIVATE apg_sgm)
endif()

if(APG_SGM_BUILD_TESTS)
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

    add_executable(test_c_api tests/test_c_api.cpp)
    target_link_libraries(test_c_api PRIVATE apg_sgm)
    add_test(NAME c_api COMMAND test_c_api)

    add_executable(test_hardening tests/test_hardening.cpp)
    target_link_libraries(test_hardening PRIVATE apg_sgm)
    add_test(NAME hardening COMMAND test_hardening)

    add_executable(c_api_smoke tests/c_api_smoke.c)
    target_link_libraries(c_api_smoke PRIVATE apg_sgm)
    add_test(NAME c_api_smoke COMMAND c_api_smoke)
endif()

# Installation & Package Export
install(TARGETS apg_sgm
    EXPORT apg_sgmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/apg_sgm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(APG_SGM_BUILD_APPS)
    install(TARGETS apg-stereo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(APG_SGM_BUILD_BENCHMARKS)
    install(TARGETS apg-benchmark RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(EXPORT apg_sgmTargets
    FILE apg_sgmTargets.cmake
    NAMESPACE apg_sgm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/apg_sgm
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/apg_sgmConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apg_sgmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/apg_sgmConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/apg_sgm
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/apg_sgmConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/apg_sgmConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/apg_sgm
)
