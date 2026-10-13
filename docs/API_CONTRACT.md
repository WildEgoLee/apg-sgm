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
3. **Disparity Ranges**:
   - `max_disparity > min_disparity`.
   - Search range may include negative disparities if configured.

---

## 2. Output Contract

1. **Disparity Representation (`Image32f disparity`)**:
   - Valid disparity values are non-negative floats in pixel units, incorporating subpixel interpolation if enabled.
   - Pixels marked invalid or occluded are assigned exactly `-1.0f`.
   - Values never evaluate to `NaN` or `+Inf` in the output disparity map.
2. **Confidence Map (`Image32f confidence`)**:
   - Value range is $[0.0, 1.0]$.
   - Higher values indicate higher matching certainty (based on uniqueness, texture, and left-right consistency).
3. **Invalidity Reasons (`Image8u1 invalid_reason`)**:
   - `0`: Valid match.
   - Non-zero bits indicate rejection cause (e.g., occlusion, low texture, left-right mismatch).
4. **Reliable Mask (`Image8u1 reliable_mask`)**:
   - Binary indicator (`0` or `1`) representing whether the pixel passed confidence and consistency thresholds.

---

## 3. Behavioral Invariants

1. **Numerical Determinism**:
   - Given the identical `PipelineConfig`, thread count, and inputs, the output disparity must be 100% bit-exact across multiple runs and between one-shot / streaming calls.
2. **Backend Equivalence**:
   - Valid disparity estimates across `Auto`, `Dense`, and `Packed` backends must agree within subpixel parity tolerances ($< 10^{-4}$ disparity difference on valid pixels, exact on invalid pixels).
3. **Memory Safety & Retention**:
   - Under default execution options, frame processing must not retain working buffers larger than the incoming frame requirements.
   - Bounded reuse buffers must strictly respect `max_retained_bytes` limits.
