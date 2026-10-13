# APG-SGM P4 Architecture & Engineering Specification

## 1. Executive Summary & Vision

APG-SGM P3 completed the CPU algorithm optimization phase, delivering a 4.09x cumulative speedup over the canonical baseline (14.63s full Middlebury 16-thread benchmark, with 100% bit-exact parity).

**P4 transformed APG-SGM from a monolithic research prototype into an embeddable, stream-oriented, resource-controlled, observable, and packageable production stereo engine.**

Key achievements across P4 milestones:
1. **Public Output Contract (`StereoResult`)**: Cleanly decoupled product consumer deliverables (`disparity`, `confidence`, `invalid_reason`, `reliable_mask`) from internal scratch buffers (`PipelineBuffers`).
2. **Explicit Memory Policy & Workspace (`StereoWorkspace`)**: Replaced hidden memory leaks and uncontrolled retention with explicit policies:
   - `MemoryPolicy::ReleasePerFrame`: Default production policy ensuring 0-byte retention after each frame.
   - `MemoryPolicy::BoundedReuse`: Steady-state capacity reuse mode for streaming pipelines with explicit `max_retained_bytes` resource guarding. Verified zero reallocations for major workspace buffers across identical-dimension frames via `allocation_count` and `reuse_count`.
   - **Pre-compute Budget Gating**: `estimate_workspace_bytes()` evaluates required memory *before* dispatching allocations, aborting immediately with `StatusCode::ResourceLimitExceeded` if budget is exceeded.
3. **Structured Status & Validation (`Status`, `StatusCode`)**: Introduced granular error reporting (`EmptyInput`, `SizeMismatch`, `InvalidDisparityRange`, `InvalidConfig`, `UnsupportedBackend`, `ResourceLimitExceeded`, `AllocationFailure`, `InternalError`) without relying on exceptions or ambiguous boolean flags.
4. **Execution Policy & Threading Control**: Disentangled algorithm hyperparameters (`PipelineConfig`) from execution options (`ExecutionOptions`). `ExecutionOptions::num_threads` dynamically governs OpenMP thread dispatch.
5. **Continuous Session & Streaming Engine (`StereoSession`)**: Reusable engine instance for processing frame sequences, maintaining running frame counts, cumulative timings, average latency, and workspace reuse.
6. **Raw-Memory Strided Interoperability (`ImageView8`, `StereoInputView`)**: Zero-copy input view accepting arbitrary row strides and memory layouts from camera SDKs and foreign runtimes.
7. **Observability, Diagnostics & Accounting (`PipelineStats`)**: Enhanced telemetry capturing per-stage timings, `peak_workspace_bytes`, `retained_workspace_bytes`, `valid_disparity_ratio`, execution parameters (`requested_threads`, `effective_threads`, `requested_backend`, `resolved_backend`, `memory_policy`), and resource instrumentation (`allocation_count`, `reuse_count`), with structured formatting via `to_string()` and `to_json()`. CLI `apg-stereo` exposes `--stats` and `--stats-json`.
8. **Production Packaging & CMake Distribution**: Standard modern CMake packaging with configurable build flags (`APG_SGM_BUILD_APPS`, `APG_SGM_BUILD_TESTS`, `APG_SGM_BUILD_BENCHMARKS`), `GNUInstallDirs`, generator expressions, target exports, `apg_sgmConfig.cmake`, `apg_sgmConfigVersion.cmake` for `find_package(apg_sgm CONFIG)`. Verified by independent external consumer project (`tests/test_cmake_package`).
9. **Pure C ABI Boundary (`include/apg_sgm/c_api.h`)**: Stable C ABI interface with opaque handles (`apg_session_t`, `apg_result_t`), extended configuration (`apg_session_options_t`, `apg_session_create_ex`), and strided image descriptors (`apg_image_view_t`). Verified by pure C99 compilation in `tests/c_api_smoke.c`.
10. **Release Hardening & Sanitizers**: Fault injection tests (huge bounds, invalid disparity ranges, invalid row strides, budget failure, fault recovery across frames). Multi-platform CI matrix covering Linux GCC, Linux Clang, Linux Clang ASan/UBSan, Linux GCC Shared Library, and Windows MSVC.
11. **Algorithmic Invariant**: 100% bit-exact output parity with P3 across all operational modes (`Fast`, `Balanced`, `HighQuality`, `Auto`, `Dense`, `Packed`).

---

## 2. P4 Architecture Hierarchy

```text
+=============================================================================+
|                                Public C ABI Layer                           |
|       apg_session_t | apg_result_t | apg_image_view_t | apg_status_code_t   |
|       apg_session_create_ex(const apg_session_options_t*)                  |
+=============================================================================+
                                      |
                                      v
+=============================================================================+
|                              Public C++ API Layer                           |
|   +-----------------------+                    +------------------------+   |
|   |    PipelineConfig     |                    |    ExecutionOptions    |   |
|   |  (algorithm params)   |                    | (memory policy/threads)|   |
|   +-----------------------+                    +------------------------+   |
|               |                                            |                |
|               +--------------------+-----------------------+                |
|                                    v                                        |
|                          +-------------------+                              |
|                          |   StereoSession   |                              |
|                          +-------------------+                              |
|                                    |                                        |
|         +--------------------------+--------------------------+             |
|         |                          |                          |             |
|         v                          v                          v             |
| +-----------------+       +-----------------+        +-----------------+    |
| | StereoInputView |       |  StereoResult   |        |  PipelineStats  |    |
| |  (ImageView8)   |       | (disparity/conf)|        | (timing/memory) |    |
| +-----------------+       +-----------------+        +-----------------+    |
+=============================================================================+
                                      |
                                      v
+=============================================================================+
|                             Core Engine Layer                               |
|   +---------------------------------------------------------------------+   |
|   |                            StereoMatcher                            |   |
|   |           (Stateless, thread-safe algorithmic execution core)       |   |
|   +---------------------------------------------------------------------+   |
|                                     |                                       |
|                                     v                                       |
|   +---------------------------------------------------------------------+   |
|   |                           StereoWorkspace                           |   |
|   |     (Controlled memory lifecycle: ReleasePerFrame / BoundedReuse)   |   |
|   |     - Pre-compute budget gating: estimate_workspace_bytes()         |   |
|   |     - Instrumentation counters: allocation_count, reuse_count       |   |
|   |                                                                     |   |
|   |   +-------------------------------------------------------------+   |   |
|   |   |                      PipelineBuffers                        |   |   |   |
|   |   |  - Census Descriptors (Gray / Gradient / Census)            |   |   |   |
|   |   |  - Cost Volumes (Dense CostVolume / PackedCostVolume)       |   |   |   |
|   |   |  - SGM Cost Accumulators (4-path / 8-path)                  |   |   |   |
|   |   |  - Prior Estimator Supports & Depth Boundary Buffers        |   |   |   |
|   |   +-------------------------------------------------------------+   |   |
|   +---------------------------------------------------------------------+   |
+=============================================================================+
```

---

## 3. Detailed Component Specifications

### 3.1 Public Result Representation (`StereoResult`)
Defined in `include/apg_sgm/result.hpp`:
- `Image32f disparity`: Floating-point disparity map (in pixels, subpixel-accurate). Invalid / occluded pixels are assigned `-1.0f`.
- `Image32f confidence`: Confidence score in range $[0.0, 1.0]$.
- `Image8u1 invalid_reason`: Granular pixel failure code (`InvalidReason::Valid`, `Occlusion`, `Mismatch`, `LowConfidence`, `OutOfRange`).
- `Image8u1 reliable_mask`: Binary indicator (`1` = valid reliable match, `0` = rejected).
- Method `clear()` releases internal matrices; `empty()` indicates absence of result data.

### 3.2 Error Model & Input Validation (`Status`, `StatusCode`)
Defined in `include/apg_sgm/status.hpp`:
```cpp
enum class StatusCode {
    Ok = 0,
    EmptyInput,
    SizeMismatch,
    InvalidDisparityRange,
    InvalidConfig,
    UnsupportedBackend,
    ResourceLimitExceeded,
    AllocationFailure,
    InternalError
};
```
- Non-throwing error model: all operations return `Status`.
- Pre-execution validation occurs via `validate_config(config)` and `validate_inputs(left, right)`.
- If input dimensions, channels, or disparity bounds are invalid, execution immediately aborts with descriptive error strings before memory allocation or compute dispatch.

### 3.3 Memory Policy & Resource Guarding (`StereoWorkspace`)
Defined in `include/apg_sgm/execution.hpp` and `include/apg_sgm/workspace.hpp`:
- `MemoryPolicy::ReleasePerFrame` (Default):
  - Scratch buffers are fully released at the end of each frame computation.
  - Guaranteed 0-byte steady-state retention across calls.
  - Prevents memory bloat across heterogeneous image streams.
- `MemoryPolicy::BoundedReuse`:
  - Retains cost volumes and intermediate buffers between consecutive frames with matching dimensions.
  - Achieves zero-reallocation steady-state performance for fixed-resolution video streaming, verified via `allocation_count` and `reuse_count`.
  - Respects `max_retained_bytes`. If estimated or retained memory exceeds the threshold, `Status(StatusCode::ResourceLimitExceeded)` is returned.
- **Pre-compute Budget Check**:
  - `estimate_workspace_bytes(w, h, cfg)` evaluates required memory before allocating any major buffer.
  - If `estimated > max_retained_bytes`, returns `ResourceLimitExceeded` immediately with 0 bytes allocated.

### 3.4 Continuous Streaming Engine (`StereoSession`)
Defined in `include/apg_sgm/session.hpp`:
- Wraps `StereoMatcher`, `StereoWorkspace`, and session-level telemetry.
- Provides `process_status(...)` and `process(...)` accepting `ImageView8` or `Image8`.
- Tracks `frame_count()`, `total_time_ms()`, and `average_time_ms()`.
- Method `reset()` flushes workspace buffers and resets metrics.

### 3.5 Strided Image Views (`ImageView8`, `StereoInputView`)
Defined in `include/apg_sgm/image_view.hpp`:
- Non-owning view wrapper supporting custom row strides (`stride_bytes >= width * channels`).
- Eliminates memory copies when passing camera frames or OpenCV `cv::Mat` / NumPy array buffers.
- Supports 1-channel grayscale and 3-channel interleaved color.
- Transparent fallback conversion via `to_image()`.

### 3.6 Observability & Telemetry (`PipelineStats`)
Defined in `include/apg_sgm/pipeline_stats.hpp`:
- Detailed stage latencies (`time_aux_ms`, `time_cost_ms`, `time_agg_ms`, `time_sgm_ms`, `time_prior_ms`, `time_conf_ms`, `time_refine_ms`, `time_post_ms`, `time_total_ms`).
- Execution environment tracking: `requested_threads`, `effective_threads`, `requested_backend`, `resolved_backend`, `memory_policy`, `allocation_count`, `reuse_count`.
- Workspace memory accounting: `peak_workspace_bytes` and `retained_workspace_bytes`.
- Reliability analytics: `valid_disparity_ratio` (percentage of pixels passing confidence and LR checks).
- Export formats: `to_string()` for human-readable diagnostic logging and `to_json()` for structured APM ingestion.
- CLI exposure in `apg-stereo` via `--stats` and `--stats-json`.

### 3.7 Pure C ABI Foundation (`c_api.h`)
Defined in `include/apg_sgm/c_api.h` and `src/c_api.cpp`:
- Exported functions using `APG_API` macro (`__declspec(dllexport)` / `__attribute__((visibility("default")))`).
- Opaque handles `apg_session_t` and `apg_result_t`.
- Extended initialization `apg_session_create_ex(const apg_session_options_t*)`.
- C-compatible strided input struct `apg_image_view_t`.
- Zero-copy result pointer accessors (`apg_result_get_disparity`, `apg_result_get_confidence`, etc.).
- Metadata queries (`apg_session_get_last_error`, `apg_session_get_allocated_bytes`, `apg_session_get_frame_count`, `apg_session_get_total_time_ms`, `apg_status_string`, `apg_version_string`).
- Verified with pure C compiler test `tests/c_api_smoke.c`.

---

## 4. Verification & Hardening Suite

The P4 test suite guarantees full behavioral and functional compliance across 13 test targets:
1. `sanity`: Basic pipeline smoke tests.
2. `synthetic`: End-to-end synthetic stereo accuracy across modes.
3. `ablation`: Micro-benchmarks across ablation settings.
4. `headers`: Self-containment test for all public C and C++ headers.
5. `api_contract`: FNV-1a fingerprint invariance asserting bit-exact P3 baseline parity.
6. `status`: Comprehensive error handling and input validation coverage.
7. `workspace`: Verification of `ReleasePerFrame`, `BoundedReuse`, and resource limits.
8. `session`: Multi-frame streaming and telemetry validation.
9. `image_view`: Strided memory views and zero-copy boundary verification.
10. `observability`: JSON/string telemetry formatting and memory tracking.
11. `c_api`: C++ wrapper test for C API lifecycle and bit-exact parity.
12. `hardening`: Fault injection, budget pre-compute gate, session recovery, and bounded reuse steady-state counters.
13. `c_api_smoke`: Pure C99 compiler test verifying clean compilation, linking, and execution.

---

## 5. CMake Distribution Specification

The library exports standard CMake package configuration:
```cmake
find_package(apg_sgm 1.0.0 REQUIRED)
target_link_libraries(my_app PRIVATE apg_sgm::apg_sgm)
```
- Build options:
  - `APG_SGM_BUILD_APPS`: Builds CLI applications (`apg-stereo`).
  - `APG_SGM_BUILD_TESTS`: Builds registered CTest test executables.
  - `APG_SGM_BUILD_BENCHMARKS`: Builds benchmarking harness (`apg-benchmark`).
- Header files installed under `${CMAKE_INSTALL_INCLUDEDIR}/apg_sgm/`.
- Targets exported into `${CMAKE_INSTALL_LIBDIR}/cmake/apg_sgm/apg_sgmTargets.cmake`.
- Export verified via independent out-of-tree consumer project (`tests/test_cmake_package`).
