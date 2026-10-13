# APG-SGM P4 Architecture & Engineering Specification

## 1. Executive Summary & Vision

APG-SGM P3 completed the CPU algorithm optimization phase, delivering a 4.09x cumulative speedup over the canonical baseline (14.63s full Middlebury 16-thread benchmark, with 100% bit-exact parity).

**P4 transformed APG-SGM from a monolithic research prototype into an embeddable, stream-oriented, resource-controlled, observable, and packageable production stereo engine.**

Key achievements across P4 milestones:
1. **Public Output Contract (`StereoResult`)**: Cleanly decoupled product consumer deliverables (`disparity`, `confidence`, `invalid_reason`, `reliable_mask`) from internal scratch buffers (`PipelineBuffers`).
2. **Explicit Memory Policy & Workspace (`StereoWorkspace`)**: Replaced hidden memory retention with explicit policies:
   - `MemoryPolicy::ReleasePerFrame`: Default production policy ensuring 0-byte retention after each frame.
   - `MemoryPolicy::BoundedReuse`: Steady-state capacity reuse mode for streaming pipelines with explicit `max_retained_bytes` resource guarding. Verified zero workspace reallocations across identical-dimension frames via `frame_allocation_count` and `frame_reuse_count` (with `allocation_count` and `reuse_count` preserved as backward-compatible aliases). Buffer capacity tracking accurately accounts for committed capacities across large $\to$ small frame transitions.
   - **Pre-compute Budget Gating**: `estimate_workspace_bytes()` uses checked 64-bit arithmetic against integer multiplication overflow and accounts for all 6 grayscale/gradient images, census descriptors, search range bounds, packed layout, cost volumes, 10 persistent `Image32f` buffers, 2 temporary WTA `Image32f` buffers (`best`, `second`), and post-processing masks. It aborts immediately with `StatusCode::ResourceLimitExceeded` if budget is exceeded before allocating.
3. **Structured Status & Validation (`Status`, `StatusCode`)**: Introduced granular error reporting (`EmptyInput`, `SizeMismatch`, `InvalidDisparityRange`, `InvalidConfig`, `UnsupportedBackend`, `ResourceLimitExceeded`, `AllocationFailure`, `InternalError`) without relying on exceptions or ambiguous boolean flags.
4. **Execution Policy & Threading Control**: Disentangled algorithm hyperparameters (`PipelineConfig`) from execution options (`ExecutionOptions`). `ExecutionOptions::num_threads` dynamically governs OpenMP thread dispatch.
5. **Continuous Session & Streaming Engine (`StereoSession`)**: Reusable engine instance for processing frame sequences, maintaining running frame counts, cumulative timings, average latency, and workspace reuse.
6. **Raw-Memory Strided Interoperability (`ImageView8`, `StereoInputView`)**: Zero-copy input view accepting arbitrary row strides and memory layouts from camera SDKs and foreign runtimes.
7. **Observability, Diagnostics & Accounting (`PipelineStats`)**: Enhanced telemetry capturing per-stage timings, `peak_workspace_bytes`, `retained_workspace_bytes`, `valid_disparity_ratio`, execution parameters (`requested_threads`, `effective_threads`, `requested_backend`, `resolved_backend`, `memory_policy`), and resource instrumentation (`frame_allocation_count`, `frame_reuse_count`), with structured formatting via `to_string()` and `to_json()`. CLI `apg-stereo` exposes `--stats` and `--stats-json`.
8. **Production Packaging & CMake Distribution**: Standard modern CMake packaging with configurable build flags (`APG_SGM_BUILD_APPS`, `APG_SGM_BUILD_TESTS`, `APG_SGM_BUILD_BENCHMARKS`), `GNUInstallDirs`, generator expressions, target exports, `apg_sgmConfig.cmake`, `apg_sgmConfigVersion.cmake` for `find_package(apg_sgm CONFIG)`. Verified by independent external consumer project (`tests/test_cmake_package`).
9. **Pure C ABI Boundary (`include/apg_sgm/c_api.h`)**: Stable C ABI interface with opaque handles (`apg_session_t`, `apg_result_t`), struct layout freezing and version validation via `uint32_t struct_size` in `apg_session_options_t`, extended initialization (`apg_session_create_ex`), and strided image descriptors (`apg_image_view_t`). Verified by pure C99 compilation in `tests/c_api_smoke.c`.
10. **Release Hardening & Sanitizers**: Fault injection tests (huge bounds, invalid disparity ranges, invalid row strides, budget failure, fault recovery across frames, capacity retention, overflow checks). Multi-platform CI matrix covering Linux GCC, Linux Clang, Linux Clang ASan/UBSan, Linux GCC Shared Library, and Windows MSVC.
11. **Algorithmic Invariant & Performance Gate**: 100% bit-exact output parity with P3 across all operational modes, verified across all 15 Middlebury scenes with +11.84% geomean speedup and 0.00% regression over baseline `f906b4d`.

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
|   |     - Instrumentation counters: frame_allocation_count, reuse_count |   |
|   |                                                                     |   |
|   |   +-------------------------------------------------------------+   |   |
|   |   |                      PipelineBuffers                        |   |   |   |
|   |   |  - Census Descriptors (Gray / Gradient / Census)            |   |   |   |
|   |   |  - Cost Volumes (Dense CostVolume / PackedCostVolume)       |   |   |   |
|   |   |  - SGM Cost Accumulators (4-path / 8-path)                  |   |   |   |
|   |   |  - Prior Estimator Supports & Depth Boundary Buffers        |   |   |   |
|   |   +-------------------------------------------------------------+   |   |
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
  - Achieves zero frame reallocation steady-state performance for fixed-resolution video streaming, verified via `frame_allocation_count` and `frame_reuse_count`.
  - Buffer capacity tracking (`allocated_bytes()`) reflects actual committed buffer capacities, preventing under-reporting during large-to-small frame transitions.
  - Respects `max_retained_bytes`. If estimated or retained memory exceeds the threshold, `Status(StatusCode::ResourceLimitExceeded)` is returned.
- **Pre-compute Budget Check**:
  - `estimate_workspace_bytes(w, h, cfg)` evaluates required memory before allocating any major buffer using checked 64-bit arithmetic to prevent overflow.
  - Accounts for all 6 grayscale/gradient images, census descriptors, search range bounds, packed layout, cost volumes, 10 persistent `Image32f` buffers, 2 temporary WTA `Image32f` buffers (`best`, `second`), and post-processing masks.
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
- Execution environment tracking: `requested_threads`, `effective_threads`, `requested_backend`, `resolved_backend`, `memory_policy`, `frame_allocation_count`, `frame_reuse_count` (with `allocation_count` / `reuse_count` aliases).
- Workspace memory accounting: `peak_workspace_bytes` and `retained_workspace_bytes`.
- Reliability analytics: `valid_disparity_ratio` (percentage of pixels passing confidence and LR checks).
- Export formats: `to_string()` for human-readable diagnostic logging and `to_json()` for structured APM ingestion.
- CLI exposure in `apg-stereo` via `--stats` and `--stats-json`.

### 3.7 Pure C ABI Foundation (`c_api.h`)
Defined in `include/apg_sgm/c_api.h` and `src/c_api.cpp`:
- Exported functions using `APG_API` macro (`__declspec(dllexport)` / `__attribute__((visibility("default")))`).
- Opaque handles `apg_session_t` and `apg_result_t`.
- Explicit struct layout validation and version freezing via `uint32_t struct_size` header in `apg_session_options_t`. Callers pass `sizeof(apg_session_options_t)` or `0`; layout mismatches are rejected by `apg_session_create_ex` with `nullptr`.
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

---

## 6. Canonical Performance Regression Gate & P4 Closure

To guarantee that the introduction of `StereoSession`, `StereoWorkspace`, strided `ImageView8`, `PipelineStats`, and C ABI layers introduced zero performance regression against the P3 final optimization baseline (`f906b4d`), a paired canonical benchmark was executed across all 15 Middlebury 2014 scenes in the identical execution environment (MSVC x64 Release, 16 threads, warmup=2, repeat=5):

| Middlebury 2014 Scene | Baseline `f906b4d` (ms) | P4 Final Candidate (ms) | Delta (ms) | Speedup / Regression | Bit-Exact Parity |
| :-------------------- | ----------------------: | ----------------------: | ---------: | -------------------: | :--------------: |
| **Adirondack**        |                   73.01 |                   70.03 |      +2.98 |               +4.26% |       PASS       |
| **ArtL**              |                   21.47 |                   19.07 |      +2.40 |              +12.59% |       PASS       |
| **Jadeplant**         |                   97.63 |                   88.22 |      +9.41 |              +10.67% |       PASS       |
| **Motorcycle**        |                   79.41 |                   67.88 |     +11.53 |              +16.99% |       PASS       |
| **MotorcycleE**       |                   77.22 |                   70.46 |      +6.76 |               +9.59% |       PASS       |
| **Piano**             |                   68.48 |                   61.79 |      +6.69 |              +10.83% |       PASS       |
| **PianoL**            |                   79.52 |                   69.12 |     +10.40 |              +15.05% |       PASS       |
| **Pipes**             |                   77.52 |                   67.54 |      +9.98 |              +14.78% |       PASS       |
| **Playroom**          |                   76.01 |                   66.55 |      +9.46 |              +14.21% |       PASS       |
| **Playtable**         |                   72.17 |                   66.46 |      +5.71 |               +8.59% |       PASS       |
| **PlaytableP**        |                   69.73 |                   64.38 |      +5.35 |               +8.31% |       PASS       |
| **Recycle**           |                   76.12 |                   68.38 |      +7.74 |              +11.32% |       PASS       |
| **Shelves**           |                   74.66 |                   66.12 |      +8.54 |              +12.92% |       PASS       |
| **Teddy**             |                   34.32 |                   30.05 |      +4.27 |              +14.21% |       PASS       |
| **Vintage**           |                   99.53 |                   87.35 |     +12.18 |              +13.94% |       PASS       |
| **TOTAL / GEOMEAN**   |             **1076.80** |              **963.40** |   **+113.40**| **1.1184x (+11.84%)** | **100% BIT-EXACT**|

### Final Technical Evaluation
1. **Runtime Performance**: Total pipeline latency improved from **1076.80 ms to 963.40 ms** (**+11.77% speedup**, **1.1184x geomean**). Zero regression across all 15 benchmark scenes (exceeding the strict $\le 2\%$ regression budget).
2. **Numerical & Parity Accuracy**: 100% bit-exact parity across all 15 scenes (EPE, Bad2.0, and D1-all identical).
3. **Memory Safety & Accounting**: Capacity-based workspace retained-memory accounting guarantees accurate reporting on dynamic stream size changes; pre-compute checked bounds prevent overflow; default production mode maintains strict 0-byte retention.
4. **C ABI Stability**: Frozen `apg_session_options_t` layout with `struct_size` version guarding and pure C99 compatibility.

**Conclusion: P4 ARCHITECTURE COMPLETE.**
