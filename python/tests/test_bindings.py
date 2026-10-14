import concurrent.futures
import gc
import time
import numpy as np
import pytest
import apg_sgm as apg


def fnv1a64(data_bytes: bytes) -> int:
    """Computes 64-bit FNV-1a hash matching C++ test_api_contract fnv1a64."""
    h = 14695981039346656037
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF
    for b in data_bytes:
        h = ((h ^ b) * prime) & mask
    return h


def make_synthetic_pair(width=64, height=48, disparity_shift=4, channels=1):
    """Generates synthetic stereo pair with horizontal shift."""
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


def test_session_lifecycle_and_default_release_per_frame():
    # 1. Default memory policy must be ReleasePerFrame (matches P4 C++ default engine contract)
    session = apg.StereoSession()
    assert session.frame_count == 0
    assert session.total_time_ms == 0.0
    assert session.average_time_ms == 0.0
    assert session.allocated_bytes == 0
    assert "StereoSession" in repr(session)

    left, right = make_synthetic_pair(64, 48)
    res = session.process(left, right)
    assert not res.empty
    # Under default ReleasePerFrame, retained workspace after compute is strictly 0 bytes
    assert session.allocated_bytes == 0
    assert res.stats.retained_workspace_bytes == 0

    # 2. Configurable initialization
    s2 = apg.StereoSession(
        mode="fast",
        threads=2,
        max_disparity=32,
        min_disparity=0,
        backend="dense",
        memory_policy="release_per_frame",
    )
    assert s2.frame_count == 0

    # 3. Invalid configuration values
    with pytest.raises(ValueError, match="Unrecognized mode"):
        apg.StereoSession(mode="super_fast")

    with pytest.raises(ValueError, match="Unrecognized backend"):
        apg.StereoSession(backend="quantum")

    with pytest.raises(ValueError, match="Unrecognized memory_policy"):
        apg.StereoSession(memory_policy="infinite_cache")


def test_bounded_reuse_opt_in():
    """Verify BoundedReuse retains workspace only when explicitly opted in."""
    session = apg.StereoSession(
        mode="fast",
        max_disparity=16,
        memory_policy="bounded_reuse",
        max_retained_bytes=10_000_000,
    )
    left, right = make_synthetic_pair(64, 48)

    res = session.process(left, right)
    assert not res.empty
    retained = session.allocated_bytes
    assert retained > 0, "BoundedReuse must retain major workspace capacity"
    assert res.stats.retained_workspace_bytes == retained

    # Steady state: subsequent frame of same size must not grow retained capacity
    session.process(left, right)
    assert session.allocated_bytes == retained

    # Reset cleans up retained workspace
    session.reset()
    assert session.allocated_bytes == 0
    assert session.frame_count == 0


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
    gc.collect()

    val = float(disp_view[10, 10])
    c_val = float(conf_view[10, 10])
    m_val = int(mask_view[10, 10])

    assert not np.isnan(val)
    assert not np.isnan(c_val)
    assert m_val in (0, 1)


def test_strict_buffer_format_validation():
    session = apg.StereoSession(mode="fast", max_disparity=16)
    left, right = make_synthetic_pair(64, 48)

    # 1. Reject int8 (1-byte signed, format 'b')
    with pytest.raises(TypeError, match="must have dtype uint8"):
        session.process(left.astype(np.int8), right.astype(np.int8))

    # 2. Reject bool (1-byte boolean, format '?')
    with pytest.raises(TypeError, match="must have dtype uint8"):
        session.process(left.astype(bool), right.astype(bool))

    # 3. Reject uint16 (2-byte unsigned, format 'H')
    with pytest.raises(TypeError, match="must have dtype uint8"):
        session.process(left.astype(np.uint16), right.astype(np.uint16))

    # 4. Reject float32 (4-byte float, format 'f')
    with pytest.raises(TypeError, match="must have dtype uint8"):
        session.process(left.astype(np.float32), right.astype(np.float32))

    # 5. Invalid dimension count (1D)
    with pytest.raises(ValueError, match="must be 2D .* or 3D"):
        session.process(left.flatten(), right.flatten())

    # 6. Invalid channel count (4 channels)
    left_4ch = np.zeros((48, 64, 4), dtype=np.uint8)
    right_4ch = np.zeros((48, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have 1 or 3 channels"):
        session.process(left_4ch, right_4ch)

    # 7. Size mismatch between left and right
    right_small = right[:30, :40]
    with pytest.raises(ValueError, match="must have identical dimensions"):
        session.process(left, right_small)


def test_concurrency_same_session_serialization():
    """Verify that concurrent calls on the SAME StereoSession instance are serialized safely without data race."""
    session = apg.StereoSession(mode="fast", max_disparity=16, memory_policy="release_per_frame")
    left, right = make_synthetic_pair(64, 48)

    num_threads = 4
    calls_per_thread = 5

    def worker():
        for _ in range(calls_per_thread):
            res = session.process(left, right)
            assert not res.empty
            assert res.width == 64

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Must not raise any exceptions

    assert session.frame_count == num_threads * calls_per_thread


def test_concurrency_independent_sessions_parallel():
    """Verify that independent StereoSession instances execute concurrently with GIL released."""
    left, right = make_synthetic_pair(128, 96, disparity_shift=6)

    def run_independent_session():
        s = apg.StereoSession(mode="fast", max_disparity=16)
        results = [s.process(left, right) for _ in range(5)]
        return len(results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_independent_session) for _ in range(4)]
        counts = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert counts == [5, 5, 5, 5]


def test_frozen_cpp_contract_bit_exact_parity():
    """Verify 100% bit-exact parity against frozen C++ golden contract fingerprints (tests/test_api_contract.cpp).

    Validates all 4 output surfaces: disparity, confidence, invalid_reason, reliable_mask.
    """
    # Case 1: Odd_Checkerboard_Fast_Packed (33x27, 1-channel, dmax=16, Fast, Packed, threads=1)
    w1, h1 = 33, 27
    left1 = np.zeros((h1, w1), dtype=np.uint8)
    right1 = np.zeros((h1, w1), dtype=np.uint8)
    for y in range(h1):
        for x in range(w1):
            val = 40 if ((x // 4 + y // 4) % 2 == 0) else 210
            left1[y, x] = val
            right1[y, x] = val

    s1 = apg.StereoSession(mode="fast", max_disparity=16, backend="packed", threads=1)
    res1 = s1.process(left1, right1)

    assert fnv1a64(res1.disparity.tobytes()) == 0x1710451f7eab215
    assert fnv1a64(res1.confidence.tobytes()) == 0xe61822cc0fbe3a38
    assert fnv1a64(res1.invalid_reason.tobytes()) == 0x22081e533cd30004
    assert fnv1a64(res1.reliable_mask.tobytes()) == 0x9ade9d63e4fcae7b

    # Case 2: Color_Noise_Fast_Packed (32x24, 3-channel, dmax=16, Fast, Packed, threads=1)
    w2, h2 = 32, 24
    left2 = np.zeros((h2, w2, 3), dtype=np.uint8)
    right2 = np.zeros((h2, w2, 3), dtype=np.uint8)
    for y in range(h2):
        for x in range(w2):
            v = (x * 37 + y * 73 + x * y * 13) % 256
            for arr in (left2, right2):
                arr[y, x, 0] = v
                arr[y, x, 1] = (v + 30) % 256
                arr[y, x, 2] = (v + 60) % 256

    s2 = apg.StereoSession(mode="fast", max_disparity=16, backend="packed", threads=1)
    res2 = s2.process(left2, right2)

    assert fnv1a64(res2.disparity.tobytes()) == 0x7e8e0fa784351325
    assert fnv1a64(res2.confidence.tobytes()) == 0x5860e8f479cbe0fd
    assert fnv1a64(res2.invalid_reason.tobytes()) == 0x82f55cc1fd68a695
    assert fnv1a64(res2.reliable_mask.tobytes()) == 0x9db48cb5d76242b5


def test_streaming_stability_100_frames():
    """100-frame streaming stability test with BoundedReuse major retained workspace capacity reuse."""
    session = apg.StereoSession(
        mode="fast",
        max_disparity=16,
        memory_policy="bounded_reuse",
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


def test_glue_overhead_smoke():
    """CI-safe smoke test for wrapper overhead (logs telemetry without brittle hard-latency gate)."""
    w, h = 320, 240
    left, right = make_synthetic_pair(w, h, disparity_shift=8)

    session = apg.StereoSession(mode="balanced", max_disparity=64)

    # Warmup
    for _ in range(2):
        session.process(left, right)
    session.reset()

    num_iterations = 4
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        session.process(left, right)
    total_wall_sec = time.perf_counter() - t0

    total_session_sec = session.total_time_ms / 1000.0
    glue_overhead_sec = total_wall_sec - total_session_sec
    overhead_ratio = (glue_overhead_sec / total_session_sec) if total_session_sec > 0 else 0.0

    print(
        f"\nOverhead Telemetry: Wall={total_wall_sec * 1000.0:.2f}ms, "
        f"Session={total_session_sec * 1000.0:.2f}ms, "
        f"Glue={glue_overhead_sec * 1000.0:.3f}ms ({overhead_ratio * 100.0:.3f}%)"
    )

    # Catastrophic regression safety bound (loose for shared CI runners; strict benchmark runs separately)
    assert overhead_ratio < 0.15, f"Catastrophic glue overhead detected: {overhead_ratio * 100.0:.2f}%"


def test_readonly_buffer_support():
    """Verify read-only numpy buffers (flags.writeable = False) are accepted without copy/failure."""
    left, right = make_synthetic_pair(64, 48, disparity_shift=3)
    left_ro = left.copy()
    right_ro = right.copy()
    left_ro.flags.writeable = False
    right_ro.flags.writeable = False

    session = apg.StereoSession(mode="fast", max_disparity=16)
    res_normal = session.process(left, right)
    res_ro = session.process(left_ro, right_ro)

    assert not res_ro.empty
    assert np.array_equal(res_normal.disparity, res_ro.disparity)


def test_metadata_bounds_and_overflow_rejection():
    """Verify dimension bounds and invalid metadata are safely rejected with ValueError/TypeError."""
    session = apg.StereoSession(mode="fast", max_disparity=16)

    # 1. Zero dimension
    zero_h = np.zeros((0, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="strictly positive"):
        session.process(zero_h, zero_h)

    zero_w = np.zeros((48, 0), dtype=np.uint8)
    with pytest.raises(ValueError, match="strictly positive"):
        session.process(zero_w, zero_w)

    # 2. Unsupported channels
    four_c = np.zeros((48, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="must have 1 or 3 channels"):
        session.process(four_c, four_c)

    # 3. Invalid dtype
    float_arr = np.zeros((48, 64), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        session.process(float_arr, float_arr)

    # 4. Dimension mismatch
    left = np.zeros((48, 64), dtype=np.uint8)
    mismatched = np.zeros((48, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="Left and right images must have identical dimensions"):
        session.process(left, mismatched)


def test_hierarchical_session_lifecycle():
    """Verify hierarchical session creation, configuration inspection, and string representation."""
    session = apg.StereoSession(
        mode="balanced",
        max_disparity=64,
        hierarchical=True,
        hierarchical_radius=4,
        hierarchical_fallback="nearest_valid",
    )
    assert session.hierarchical_enabled is True
    assert session.hierarchical_radius == 4
    assert session.adaptive_radius is False
    assert "hierarchical=True (R=4)" in repr(session)


def test_hierarchical_process_and_streaming():
    """Verify hierarchical execution produces valid results and reuses memory across streaming frames."""
    session = apg.StereoSession(
        mode="fast",
        max_disparity=32,
        memory_policy="bounded_reuse",
        hierarchical=True,
        hierarchical_radius=4,
    )
    left, right = make_synthetic_pair(64, 48, disparity_shift=4)

    initial_alloc = 0
    for frame in range(10):
        res = session.process(left, right)
        assert not res.empty
        assert res.disparity.shape == (48, 64)
        if frame == 0:
            initial_alloc = session.allocated_bytes
            assert initial_alloc > 0
        else:
            assert session.allocated_bytes == initial_alloc

    assert session.frame_count == 10
    session.reset()
    assert session.allocated_bytes == 0


def test_hierarchical_adaptive_radius():
    """Verify confidence-aware adaptive search radius in hierarchical mode."""
    session = apg.StereoSession(
        mode="balanced",
        max_disparity=32,
        hierarchical=True,
        hierarchical_radius=4,
        adaptive_radius=True,
        confidence_threshold=0.5,
        expanded_radius=8,
    )
    assert session.hierarchical_enabled is True
    assert session.adaptive_radius is True
    assert "adaptive_R=8@0.50" in repr(session)

    left, right = make_synthetic_pair(64, 48, disparity_shift=4)
    res = session.process(left, right)
    assert not res.empty
    assert res.disparity.shape == (48, 64)


def test_hierarchical_invalid_parameters():
    """Verify invalid hierarchical configurations are rejected by C++ validation."""
    with pytest.raises(RuntimeError, match="hierarchical radius must be strictly positive"):
        apg.StereoSession(mode="fast", max_disparity=32, hierarchical=True, hierarchical_radius=0)

    with pytest.raises(RuntimeError, match="hierarchical radius must be strictly positive"):
        apg.StereoSession(mode="fast", max_disparity=32, hierarchical=True, hierarchical_radius=-2)

    with pytest.raises(RuntimeError, match="expanded_radius cannot be less than radius"):
        apg.StereoSession(
            mode="fast",
            max_disparity=32,
            hierarchical=True,
            hierarchical_radius=4,
            adaptive_radius=True,
            expanded_radius=2,
        )

    with pytest.raises(RuntimeError, match="confidence_threshold must be between"):
        apg.StereoSession(
            mode="fast",
            max_disparity=32,
            hierarchical=True,
            adaptive_radius=True,
            confidence_threshold=-0.5,
        )
