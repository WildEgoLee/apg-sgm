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
   - Under `MemoryPolicy::BoundedReuse`, memory retained between frames with identical dimensions must not trigger reallocation, and must never exceed `max_retained_bytes`.

---

## 4. API Stability & Compatibility

1. **C++ API**:
   - Preferred modern API: `StereoSession::process_status(left_view, right_view, result)` and `StereoMatcher::compute_status(left, right, result, workspace)`.
   - Legacy API: `StereoMatcher::compute(left, right, disparity, confidence, buffers)` remains available for backward compatibility.
2. **C ABI**:
   - Exported under `include/apg_sgm/c_api.h` with symbol export macro `APG_API`.
   - ABI stability guaranteed across minor releases through opaque handle design (`apg_session_t`, `apg_result_t`).
