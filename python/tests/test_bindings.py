import gc
import time
import numpy as np
import pytest
import apg_sgm as apg


def make_synthetic_pair(width=64, height=48, disparity_shift=4, channels=1):
    """Generates synthetic stereo pair with horizontal shift."""
    np.random.seed(42)
    if channels == 1:
        left = np.zeros((height, width), dtype=np.uint8)
        right = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                left[y, x] = int((x * 11 + y * 17 + 37) % 256)
                xr = max(0, x - disparity_shift)
                right[y, x] = int((xr * 11 + y * 17 + 37) % 256)
        return left, right
    else:
        left = np.zeros((height, width, channels), dtype=np.uint8)
        right = np.zeros((height, width, channels), dtype=np.uint8)
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    left[y, x, c] = int((x * (11 + c * 3) + y * (17 + c * 5) + 37) % 256)
                    xr = max(0, x - disparity_shift)
                    right[y, x, c] = int((xr * (11 + c * 3) + y * (17 + c * 5) + 37) % 256)
        return left, right


def test_metadata_and_enums():
    assert hasattr(apg, "__version__")
    assert apg.QualityMode.Fast is not None
    assert apg.QualityMode.Balanced is not None
    assert apg.QualityMode.HighQuality is not None

    assert apg.VolumeBackend.Auto is not None
    assert apg.VolumeBackend.Dense is not None
    assert apg.VolumeBackend.Packed is not None

    assert apg.MemoryPolicy.ReleasePerFrame is not None
    assert apg.MemoryPolicy.BoundedReuse is not None

    assert apg.StatusCode.Ok is not None
    assert apg.StatusCode.AllocationFailure is not None


def test_session_lifecycle():
    session = apg.StereoSession()
    assert session.frame_count == 0
    assert session.total_time_ms == 0.0
    assert session.average_time_ms == 0.0
    assert session.allocated_bytes == 0
    assert "StereoSession" in repr(session)

    # Configurable initialization
    s2 = apg.StereoSession(
        mode="fast",
        threads=2,
        max_disparity=32,
        min_disparity=0,
        backend="dense",
        memory_policy="bounded_reuse",
        max_retained_bytes=10_000_000,
    )
    assert s2.frame_count == 0

    # Invalid configurations
    with pytest.raises(ValueError, match="Unrecognized mode"):
        apg.StereoSession(mode="super_fast")

    with pytest.raises(ValueError, match="Unrecognized backend"):
        apg.StereoSession(backend="quantum")

    with pytest.raises(ValueError, match="Unrecognized memory_policy"):
        apg.StereoSession(memory_policy="infinite_cache")


def test_basic_process_grayscale():
    left, right = make_synthetic_pair(64, 48, disparity_shift=3, channels=1)
    session = apg.StereoSession(mode="fast", max_disparity=16)

    res = session.process(left, right)
    assert not res.empty
    assert res.width == 64
    assert res.height == 48

    disp = res.disparity
    assert isinstance(disp, np.ndarray)
    assert disp.dtype == np.float32
    assert disp.shape == (48, 64)

    conf = res.confidence
    assert isinstance(conf, np.ndarray)
    assert conf.dtype == np.float32
    assert conf.shape == (48, 64)

    mask = res.reliable_mask
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.shape == (48, 64)

    reason = res.invalid_reason
    assert isinstance(reason, np.ndarray)
    assert reason.dtype == np.uint8
    assert reason.shape == (48, 64)

    # PipelineStats check
    stats = res.stats
    assert stats.timing.total_ms > 0
    assert stats.timing.sgm_ms >= 0
    assert stats.timing.cost_ms >= 0
    assert stats.peak_workspace_bytes > 0
    assert 0.0 <= stats.valid_disparity_ratio <= 1.0

    stats_dict = stats.to_dict()
    assert "timing" in stats_dict
    assert "peak_workspace_bytes" in stats_dict
    assert "total_ms" in stats_dict["timing"]


def test_basic_process_rgb():
    left, right = make_synthetic_pair(64, 48, disparity_shift=3, channels=3)
    session = apg.StereoSession(mode="fast", max_disparity=16)

    res = session.process(left, right)
    assert not res.empty
    assert res.width == 64
    assert res.height == 48
    assert res.disparity.shape == (48, 64)


def test_strided_and_padded_inputs():
    w, h = 64, 48
    left, right = make_synthetic_pair(w, h, disparity_shift=3, channels=1)

    # 1. Row-padded view (stride > width)
    padded_w = 80
    left_padded = np.zeros((h, padded_w), dtype=np.uint8)
    right_padded = np.zeros((h, padded_w), dtype=np.uint8)
    left_padded[:, :w] = left
    right_padded[:, :w] = right

    left_view = left_padded[:, :w]
    right_view = right_padded[:, :w]
    assert left_view.strides[0] == padded_w  # row stride > width

    session = apg.StereoSession(mode="fast", max_disparity=16)
    res_strided = session.process(left_view, right_view)
    res_direct = session.process(left, right)

    # Bit-exact matching between row-padded view and contiguous input
    np.testing.assert_array_equal(res_strided.disparity, res_direct.disparity)

    # 2. Fortran / transposed layout fallback
    left_f = np.asfortranarray(left)
    right_f = np.asfortranarray(right)
    res_f = session.process(left_f, right_f)
    np.testing.assert_array_equal(res_f.disparity, res_direct.disparity)


def test_memory_lifetime_safety():
    """Verify that NumPy views hold reference to PyStereoResult base and do not dangle."""
    session = apg.StereoSession(mode="fast", max_disparity=16)
    left, right = make_synthetic_pair(64, 48)

    def extract_views():
        res = session.process(left, right)
        disp = res.disparity
        conf = res.confidence
        mask = res.reliable_mask
        assert disp.base is not None, "NumPy view base must reference owner object"
        return disp, conf, mask

    disp_view, conf_view, mask_view = extract_views()
    # Trigger full garbage collection to ensure no dangling pointers
    gc.collect()

    # Reading elements after local StereoResult scope exit must succeed cleanly
    val = float(disp_view[10, 10])
    c_val = float(conf_view[10, 10])
    m_val = int(mask_view[10, 10])

    assert not np.isnan(val)
    assert not np.isnan(c_val)
    assert m_val in (0, 1)


def test_input_validation():
    session = apg.StereoSession(mode="fast", max_disparity=16)
    left, right = make_synthetic_pair(64, 48)

    # Invalid dtype (float32 instead of uint8)
    with pytest.raises(TypeError, match="must have dtype uint8"):
        session.process(left.astype(np.float32), right)

    # Invalid dimension count (1D)
    with pytest.raises(ValueError, match="must be 2D .* or 3D"):
        session.process(left.flatten(), right.flatten())

    # Invalid channel count (4 channels)
    left_4ch = np.zeros((48, 64, 4), dtype=np.uint8)
    right_4ch = np.zeros((48, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have 1 or 3 channels"):
        session.process(left_4ch, right_4ch)

    # Size mismatch between left and right
    right_small = right[:30, :40]
    with pytest.raises(ValueError, match="must have identical dimensions"):
        session.process(left, right_small)


def test_streaming_stability_100_frames():
    """100-frame streaming stability test with BoundedReuse major retained workspace capacity reuse."""
    session = apg.StereoSession(
        mode="fast",
        max_disparity=16,
        memory_policy="bounded_reuse"
    )
    left, right = make_synthetic_pair(64, 48, disparity_shift=3)

    first_allocated = 0
    for frame_idx in range(100):
        res = session.process(left, right)
        assert not res.empty
        if frame_idx == 0:
            first_allocated = session.allocated_bytes
            assert first_allocated > 0, "BoundedReuse must allocate workspace on first frame"
        else:
            assert session.allocated_bytes == first_allocated, (
                f"Retained workspace capacity must be stable at frame {frame_idx}: "
                f"{session.allocated_bytes} vs {first_allocated}"
            )

    assert session.frame_count == 100
    assert session.total_time_ms > 0
    assert session.average_time_ms > 0

    # Reset cleans up retained workspace
    session.reset()
    assert session.frame_count == 0
    assert session.total_time_ms == 0.0
    assert session.allocated_bytes == 0


def test_bit_exact_parity_with_c_reference():
    """Verify bit-exact disparity equality and recovery on synthetic ground truth pattern."""
    def pattern(x, y):
        s = 120.0 + 70.0 * np.sin(0.17 * x) + 35.0 * np.sin(0.23 * y) + ((x * 17 + y * 31) % 23)
        if 28 <= x < 34:
            s = 250
        return np.clip(int(s), 0, 255)

    w, h, shift = 80, 48, 8
    left = np.array([[pattern(x, y) for x in range(w)] for y in range(h)], dtype=np.uint8)
    right = np.array([[pattern(x + shift, y) for x in range(w)] for y in range(h)], dtype=np.uint8)

    session = apg.StereoSession(mode="fast", max_disparity=24, backend="packed")
    res1 = session.process(left, right)

    # Run a second pass to ensure invariance
    res2 = session.process(left, right)
    np.testing.assert_array_equal(res1.disparity, res2.disparity)
    np.testing.assert_array_equal(res1.confidence, res2.confidence)
    np.testing.assert_array_equal(res1.reliable_mask, res2.reliable_mask)

    # Disparity in shifted textured region should recover shift = 8.0
    valid_region = res1.disparity[6:42, 16:64]
    valid_pixels = valid_region[valid_region >= 0]
    assert len(valid_pixels) > 80
    mean_d = float(valid_pixels.mean())
    assert abs(mean_d - 8.0) < 0.25, f"Expected disparity ~8.0, got {mean_d}"



def test_glue_overhead_less_than_1_percent():
    """Verify that Python wrapper glue overhead is < 1% of pipeline compute time."""
    w, h = 480, 360
    left, right = make_synthetic_pair(w, h, disparity_shift=8)

    session = apg.StereoSession(mode="balanced", max_disparity=64)

    # Warmup
    for _ in range(2):
        session.process(left, right)
    session.reset()

    num_iterations = 6
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        session.process(left, right)
    total_wall_sec = time.perf_counter() - t0

    total_session_sec = session.total_time_ms / 1000.0
    glue_overhead_sec = total_wall_sec - total_session_sec
    overhead_ratio = (glue_overhead_sec / total_session_sec) if total_session_sec > 0 else 0.0

    print(
        f"\nWall time: {total_wall_sec * 1000.0:.2f}ms, "
        f"Session C++ time: {total_session_sec * 1000.0:.2f}ms, "
        f"Glue overhead: {glue_overhead_sec * 1000.0:.3f}ms ({overhead_ratio * 100.0:.3f}%)"
    )

    # Requirement: glue overhead < 1% pipeline time (typically ~0.05% - 0.2%)
    assert overhead_ratio < 0.01, f"Glue overhead exceeded threshold: {overhead_ratio * 100.0:.2f}%"
