#!/usr/bin/env python3
"""Interleaved matched timing test to verify runtime overhead of Candidate C vs Baseline A."""

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
BIN = ROOT / "build" / ("codex-vs2022/Release/apg-benchmark.exe" if os.name == "nt" else "codex-linux-ninja/apg-benchmark")
RESULTS = ROOT / "results" / "p5_6_study"


def run_pass(name: str, method: str, gap: int, hierarchical: bool, repeat: int = 5) -> float:
    out_csv = RESULTS / f"matched_{name}.csv"
    cmd = [
        str(BIN),
        "--manifest", str(MANIFEST),
        "--modes", "F",
        "--backend", "packed",
        "--output", str(out_csv),
        "--warmup", "2",
        "--repeat", str(repeat),
        "--inpaint-method", method,
        "--inpaint-gap", str(gap),
    ]
    if hierarchical:
        cmd.extend(["--hierarchical", "--hierarchical-radius", "4"])

    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Run failed for {name}")

    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tot_time = sum(float(r["time_total_ms"]) for r in reader)
    return tot_time


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("P5.6 Matched Performance Gate Verification (Interleaved Runs)")
    print("Baseline A: legacy (gap=32) vs Candidate C: shadow_reclass (gap=48)")
    print("=" * 80)

    print("\n[Lane L] Interleaved execution (A1, C1, A2, C2, A3, C3)...")
    t_a1 = run_pass("lane_l_A1", "legacy", 32, False)
    t_c1 = run_pass("lane_l_C1", "shadow_reclass", 48, False)
    t_a2 = run_pass("lane_l_A2", "legacy", 32, False)
    t_c2 = run_pass("lane_l_C2", "shadow_reclass", 48, False)
    t_a3 = run_pass("lane_l_A3", "legacy", 32, False)
    t_c3 = run_pass("lane_l_C3", "shadow_reclass", 48, False)

    mean_a_l = (t_a1 + t_a2 + t_a3) / 3.0
    mean_c_l = (t_c1 + t_c2 + t_c3) / 3.0
    overhead_l = (mean_c_l - mean_a_l) / mean_a_l * 100.0

    print(f"  Baseline A runs: {t_a1:.1f} ms, {t_a2:.1f} ms, {t_a3:.1f} ms -> Mean: {mean_a_l:.1f} ms")
    print(f"  Candidate C runs: {t_c1:.1f} ms, {t_c2:.1f} ms, {t_c3:.1f} ms -> Mean: {mean_c_l:.1f} ms")
    print(f"  Lane L Overhead: {overhead_l:+.2f}% (Safety Gate: <= 5.0%) -> {'PASS' if overhead_l <= 5.0 else 'FAIL'}")

    print("\n[Lane H] Interleaved execution (A1, C1, A2, C2, A3, C3)...")
    t_a1_h = run_pass("lane_h_A1", "legacy", 32, True)
    t_c1_h = run_pass("lane_h_C1", "shadow_reclass", 48, True)
    t_a2_h = run_pass("lane_h_A2", "legacy", 32, True)
    t_c2_h = run_pass("lane_h_C2", "shadow_reclass", 48, True)
    t_a3_h = run_pass("lane_h_A3", "legacy", 32, True)
    t_c3_h = run_pass("lane_h_C3", "shadow_reclass", 48, True)

    mean_a_h = (t_a1_h + t_a2_h + t_a3_h) / 3.0
    mean_c_h = (t_c1_h + t_c2_h + t_c3_h) / 3.0
    overhead_h = (mean_c_h - mean_a_h) / mean_a_h * 100.0

    print(f"  Baseline A runs: {t_a1_h:.1f} ms, {t_a2_h:.1f} ms, {t_a3_h:.1f} ms -> Mean: {mean_a_h:.1f} ms")
    print(f"  Candidate C runs: {t_c1_h:.1f} ms, {t_c2_h:.1f} ms, {t_c3_h:.1f} ms -> Mean: {mean_c_h:.1f} ms")
    print(f"  Lane H Overhead: {overhead_h:+.2f}% (Safety Gate: <= 5.0%) -> {'PASS' if overhead_h <= 5.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
