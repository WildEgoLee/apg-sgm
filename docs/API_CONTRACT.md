# APG-SGM API Contract & Behavioral Invariants

## 1. Input Contract

1. **Rectification Requirement**:
   - Both `left` and `right` input images must be pre-rectified.
   - Epipolar lines are assumed strictly horizontal along scanlines ($y_{\text{left}} = y_{\text{right}}$).
   - Corresponding points in the right image satisfy $x_{\text{right}} = x_{\text{left}} - d$.
2. **Dimensions & Formats**:
   - `left.width() == right.width()` and `left.height() == right.height()`.
   - Dimensions must be non-zero positive integers ($W > 0, H > 0$).
   - Formats supported: 8-bit unsigned grayscale (1 channel) or interleaved color (3 channels, RGB/BGR).
   - Strided inputs (`ImageView8`, `apg_image_view_t`) must satisfy `stride >= width * channels`.
3. **Disparity Ranges**:
   - `max_disparity > min_disparity`.
   - Canonical setup assumes `min_disparity == 0` and `max_disparity` divisible by 8 or 16 for AVX2 SIMD alignment.

---

## 2. Output Contract

1. **Disparity Map (`Image32f disparity` / `apg_result_get_disparity`)**:
   - Valid disparity values are non-negative floats in pixel units, incorporating subpixel interpolation if enabled.
   - Pixels marked invalid or occluded are assigned exactly `-1.0f`.
   - Values never evaluate to `NaN` or `+Inf` in the output disparity map.
2. **Confidence Map (`Image32f confidence` / `apg_result_get_confidence`)**:
   - Value range is $[0.0, 1.0]$.
   - Higher values indicate higher matching certainty (based on uniqueness, texture, and left-right consistency).
3. **Invalidity Reasons (`Image8u1 invalid_reason` / `apg_result_get_invalid_reason`)**:
   - `0`: Valid match (`InvalidReason::Valid`).
   - `1`: Occlusion (`InvalidReason::Occlusion`).
   - `2`: Mismatch (`InvalidReason::Mismatch`).
   - `3`: Low Confidence (`InvalidReason::LowConfidence`).
   - `4`: Out of Range (`InvalidReason::OutOfRange`).
4. **Reliable Mask (`Image8u1 reliable_mask` / `apg_result_get_reliable_mask`)**:
   - Binary indicator (`0` or `1`) representing whether the pixel passed confidence and consistency thresholds.

---

## 3. Behavioral Invariants

1. **Numerical Determinism**:
   - Given the identical `PipelineConfig`, thread count, and inputs, the output disparity must be 100% bit-exact across multiple runs and between one-shot / streaming calls.
   - P4 implementation maintains 100% bit-exact output parity with the P3 optimization baseline (`f906b4d`).
2. **Backend Equivalence**:
   - Valid disparity estimates across `Auto`, `Dense`, and `Packed` backends must agree within subpixel parity tolerances ($< 10^{-4}$ disparity difference on valid pixels, exact on invalid pixels).
3. **Memory Safety & Retention**:
   - Under `MemoryPolicy::ReleasePerFrame` (default), memory retained in `StereoWorkspace` after frame processing must be exactly 0 bytes.
   - Under `MemoryPolicy::BoundedReuse`, major retained workspace capacities are reused across compatible frames; `frame_reuse_count` tracks frame-level reuse epochs, not individual heap-allocation events. Workspace capacity tracking reflects actual committed buffer capacities (`allocated_bytes()`), ensuring that on large $\to$ small frame transitions, retained memory is never under-reported.
   - Pre-compute budget gating: `estimate_workspace_bytes()` uses checked arithmetic against integer multiplication overflow, accounting for all 6 grayscale/gradient images, census descriptors, search range bounds, packed layout and cost volumes, 10 persistent `Image32f` buffers, 2 temporary WTA `Image32f` buffers (`best`, `second`), and post-processing masks. It evaluates memory demand prior to allocation and rejects over-budget requests with `StatusCode::ResourceLimitExceeded`.
4. **Execution Options & Threading**:
   - `ExecutionOptions::num_threads` takes precedence over `PipelineConfig::num_threads`. When set to 0, it falls back to config or OpenMP defaults.

---

## 4. API Stability & Compatibility

1. **C++ API**:
   - Preferred modern API: `StereoSession::process_status(left_view, right_view, result)` and `StereoMatcher::compute_status(left, right, result, workspace)`.
   - Legacy API: `StereoMatcher::compute(left, right, disparity, confidence, buffers)` remains available for backward compatibility.
   - Counters: `frame_allocation_count()` and `frame_reuse_count()` track frame-level allocation epochs, with `allocation_count()` and `reuse_count()` preserved as backward-compatible aliases.
2. **C ABI**:
   - Exported under `include/apg_sgm/c_api.h` with symbol export macro `APG_API`.
   - ABI stability guaranteed across minor releases through opaque handle design (`apg_session_t`, `apg_result_t`).
   - Extended options structure `apg_session_options_t` includes `uint32_t struct_size` at the head for explicit struct layout versioning and validation. Callers pass `sizeof(apg_session_options_t)`. `0` is accepted as a legacy initialization convention for callers built against the current struct layout; it does not provide binary compatibility with pre-struct_size layouts. Layout mismatches are rejected by `apg_session_create_ex` with `nullptr`. Struct layout is frozen for 1.x.
   - Extended initialization `apg_session_create_ex` provides full C control over memory policy, budget, threads, backend, and disparity range.
   - Verified pure C99 compatibility via `c_api_smoke.c`.

---

## 5. Python API Contract (`apg_sgm`)

1. **Header Boundary**:
   - The binding layer (`python/src/binding.cpp`) only includes public headers under `include/apg_sgm/`.
   - Zero inclusion of internal headers (`#include "src/..."`) is strictly enforced.
2. **Buffer Protocol & Strict Type Checking**:
   - Accepts contiguous or row-strided (pitch-padded) NumPy `uint8` arrays of shape `(H, W)` or `(H, W, 3)`.
   - Enforces strict dtype verification (`itemsize == 1` and format `'B'`); non-uint8 dtypes (`int8`, `bool`, `uint16`, `float32`) are immediately rejected with `TypeError`.
   - Standard C-order and row-padded strided arrays are mapped directly to `apg::ImageView8` without copy. Non-standard/transposed views safely fall back to an internal contiguous buffer copy.
3. **Memory Ownership & Retention Defaults**:
   - Output views (`disparity`, `confidence`, `reliable_mask`, `invalid_reason`) return NumPy arrays whose `base` pointer references the parent `StereoResult` instance, preventing dangling pointers upon local reference exit.
   - Default memory policy is `MemoryPolicy::ReleasePerFrame`, preserving P4 baseline zero retained footprint (`allocated_bytes() == 0`). `BoundedReuse` is strictly opt-in, and setting non-zero `max_retained_bytes` is recommended in production streaming pipelines.
4. **Concurrency & Thread Safety**:
   - Within the same `StereoSession` instance, execution is serialized via internal mutex to prevent workspace data races.
   - GIL is released during compute execution (`py::gil_scoped_release`), enabling parallel compute across independent `StereoSession` instances.
5. **Build Model & CPU Dispatch Invariance**:
   - Built via `scikit-build-core` linking the primary CMake `apg_sgm` target, avoiding global compiler AVX2 flags on generic translation units and preserving the runtime CPU dispatch architecture.
