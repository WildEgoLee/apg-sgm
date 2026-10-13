#!/usr/bin/env python3
"""Dedicated wrapper glue overhead benchmark for APG-SGM Python bindings."""

import time
import numpy as np
import apg_sgm as apg


def benchmark_glue_overhead(
    width: int = 480,
    height: int = 360,
    max_disparity: int = 64,
    warmup: int = 3,
    repeat: int = 25,
    gate_threshold: float = 0.01,
) -> bool:
    print("=" * 70)
    print("          APG-SGM PYTHON WRAPPER GLUE OVERHEAD BENCHMARK")
    print("=" * 70)
    print(f"Resolution:     {width}x{height}")
    print(f"Max Disparity:  {max_disparity}")
    print(f"Warmup / Repeat:{warmup} / {repeat}")
    print(f"Gate Threshold: < {gate_threshold * 100:.1f}%\n")

    # Generate synthetic input
    left = np.zeros((height, width), dtype=np.uint8)
    right = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            left[y, x] = int((x * 11 + y * 17 + 37) % 256)
            xr = max(0, x - 8)
            right[y, x] = int((xr * 11 + y * 17 + 37) % 256)

    session = apg.StereoSession(mode="balanced", max_disparity=max_disparity)

    # Warmup
    for _ in range(warmup):
        session.process(left, right)
    session.reset()

    # Timed runs
    wall_times_ms = []
    session_times_ms = []

    for _ in range(repeat):
        t0 = time.perf_counter()
        session.process(left, right)
        t_wall = (time.perf_counter() - t0) * 1000.0
        t_session = session.total_time_ms - (sum(session_times_ms))
        wall_times_ms.append(t_wall)
        session_times_ms.append(t_session)

    total_wall = sum(wall_times_ms)
    total_session = sum(session_times_ms)
    overhead_ms = total_wall - total_session
    overhead_ratio = overhead_ms / total_session if total_session > 0 else 0.0

    median_wall = float(np.median(wall_times_ms))
    median_session = float(np.median(session_times_ms))
    median_overhead_ratio = (median_wall - median_session) / median_session

    print(f"Total Wall Time:    {total_wall:.2f} ms")
    print(f"Total Session Time: {total_session:.2f} ms")
    print(f"Mean Glue Overhead: {overhead_ms / repeat:.3f} ms / frame ({overhead_ratio * 100:.3f}%)")
    print(f"Median Glue Overhead: {median_overhead_ratio * 100:.3f}%")

    passed = overhead_ratio < gate_threshold
    print(f"\nResult: {'PASS (GO)' if passed else 'FAIL (NO-GO)'} (Gate: < {gate_threshold * 100:.1f}%)")
    print("=" * 70)
    return passed


if __name__ == "__main__":
    import sys
    success = benchmark_glue_overhead()
    sys.exit(0 if success else 1)
