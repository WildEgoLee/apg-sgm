#!/usr/bin/env python3
"""Interleaved matched timing test to verify runtime speedup / overhead of QualityV2 vs Baseline A.

Follows the interleaved protocol: A1 -> Q1 -> A2 -> Q2 -> A3 -> Q3
under identical binary, session, thread count, backend, warmup, and repeat settings.
"""

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
BIN = ROOT / "build" / ("codex-vs2022/Release/apg-benchmark.exe" if os.name == "nt" else "codex-linux-ninja/apg-benchmark")
RESULTS = ROOT / "results" / "p5_7_study"


def run_pass(name: str, is_quality_v2: bool, repeat: int = 5, warmup: int = 2) -> float:
    out_csv = RESULTS / f"matched_{name}.csv"
    if out_csv.exists():
        out_csv.unlink()

    cmd = [
        str(BIN),
        "--manifest", str(MANIFEST),
        "--backend", "packed",
        "--output", str(out_csv),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ]

    if is_quality_v2:
        cmd.extend(["--preset", "quality_v2"])
    else:
        cmd.extend([
            "--modes", "F",
            "--inpaint-method", "legacy",
            "--inpaint-gap", "32",
        ])

    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Run failed for {name}")

    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tot_time = sum(float(r["time_total_ms"]) for r in reader)
    return tot_time


def main():
    if not BIN.exists():
        print(f"Error: binary not found at {BIN}")
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("P5.7 Matched Runtime Speedup / Overhead Verification (Interleaved Protocol)")
    print(f"Binary: {BIN}")
    print(f"Baseline A: Mode F Legacy (gap=32)")
    print(f"Profile Q:  QualityV2 Preset (Hierarchical R=4 + Shadow Reclass gap=48)")
    print("Protocol:   Interleaved A1 -> Q1 -> A2 -> Q2 -> A3 -> Q3 (warmup=2, repeat=5)")
    print("=" * 80)

    print("\nExecuting interleaved passes...")
    print("  Pass 1: A1...", flush=True)
    t_a1 = run_pass("quality_v2_A1", is_quality_v2=False)
    print(f"          A1 completed: {t_a1:.1f} ms", flush=True)

    print("  Pass 1: Q1...", flush=True)
    t_q1 = run_pass("quality_v2_Q1", is_quality_v2=True)
    print(f"          Q1 completed: {t_q1:.1f} ms", flush=True)

    print("  Pass 2: A2...", flush=True)
    t_a2 = run_pass("quality_v2_A2", is_quality_v2=False)
    print(f"          A2 completed: {t_a2:.1f} ms", flush=True)

    print("  Pass 2: Q2...", flush=True)
    t_q2 = run_pass("quality_v2_Q2", is_quality_v2=True)
    print(f"          Q2 completed: {t_q2:.1f} ms", flush=True)

    print("  Pass 3: A3...", flush=True)
    t_a3 = run_pass("quality_v2_A3", is_quality_v2=False)
    print(f"          A3 completed: {t_a3:.1f} ms", flush=True)

    print("  Pass 3: Q3...", flush=True)
    t_q3 = run_pass("quality_v2_Q3", is_quality_v2=True)
    print(f"          Q3 completed: {t_q3:.1f} ms", flush=True)

    a_runs = [t_a1, t_a2, t_a3]
    q_runs = [t_q1, t_q2, t_q3]

    mean_a = statistics.mean(a_runs)
    median_a = statistics.median(a_runs)
    mean_q = statistics.mean(q_runs)
    median_q = statistics.median(q_runs)

    overhead_mean = (mean_q - mean_a) / mean_a * 100.0
    speedup_mean = mean_a / mean_q
    overhead_median = (median_q - median_a) / median_a * 100.0
    speedup_median = median_a / median_q

    # Gate: Speedup >= 1.0x (Overhead <= 15.0%, QualityV2 aims for net speedup >= 1.10x)
    passed = (overhead_median <= 15.0) and (speedup_median >= 1.0)

    print("\n" + "=" * 80)
    print("MATCHED RUNTIME SUMMARY")
    print("=" * 80)
    print(f"  Baseline A runs (15 scenes total): {t_a1:.1f} ms, {t_a2:.1f} ms, {t_a3:.1f} ms")
    print(f"    -> Mean:   {mean_a:.1f} ms")
    print(f"    -> Median: {median_a:.1f} ms")
    print(f"  QualityV2 Q runs (15 scenes total): {t_q1:.1f} ms, {t_q2:.1f} ms, {t_q3:.1f} ms")
    print(f"    -> Mean:   {mean_q:.1f} ms")
    print(f"    -> Median: {median_q:.1f} ms")
    print("-" * 80)
    print(f"  Speedup (Mean):   {speedup_mean:.2f}x (Overhead: {overhead_mean:+.2f}%)")
    print(f"  Speedup (Median): {speedup_median:.2f}x (Overhead: {overhead_median:+.2f}%)")
    print(f"  Latency Gate:     Overhead <= 15.0% & Speedup >= 1.0x -> {'PASS' if passed else 'FAIL'}")
    print("=" * 80)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
