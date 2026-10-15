#!/usr/bin/env python3
"""Interleaved matched timing test to verify zero regression in fast_legacy path."""

import subprocess
import os
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
BIN = ROOT / "build" / ("codex-vs2022/Release/apg-benchmark.exe" if os.name == "nt" else "codex-linux-ninja/apg-benchmark")
RESULTS = ROOT / "results" / "p5_5_study"

def run_pass(name: str, hierarchical: bool, repeat: int = 3) -> float:
    out_csv = RESULTS / f"matched_{name}.csv"
    cmd = [
        str(BIN),
        "--manifest", str(MANIFEST),
        "--modes", "F",
        "--backend", "packed",
        "--output", str(out_csv),
        "--warmup", "1",
        "--repeat", str(repeat),
        "--subpixel-method", "legacy",
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
    print("Running interleaved matched runs for Lane L (Full Range Mode F)...")
    # Interleaved: L1, L2, L3, L4
    t_l1 = run_pass("lane_l_run1", False)
    t_l2 = run_pass("lane_l_run2", False)
    t_l3 = run_pass("lane_l_run3", False)
    t_l4 = run_pass("lane_l_run4", False)
    print(f"Lane L runs: {t_l1:.2f} ms, {t_l2:.2f} ms, {t_l3:.2f} ms, {t_l4:.2f} ms")
    print(f"Lane L mean: {(t_l1+t_l2+t_l3+t_l4)/4.0:.2f} ms")

    print("\nRunning interleaved matched runs for Lane H (Hierarchical Mode F R=4)...")
    t_h1 = run_pass("lane_h_run1", True)
    t_h2 = run_pass("lane_h_run2", True)
    t_h3 = run_pass("lane_h_run3", True)
    t_h4 = run_pass("lane_h_run4", True)
    print(f"Lane H runs: {t_h1:.2f} ms, {t_h2:.2f} ms, {t_h3:.2f} ms, {t_h4:.2f} ms")
    print(f"Lane H mean: {(t_h1+t_h2+t_h3+t_h4)/4.0:.2f} ms")

if __name__ == "__main__":
    main()
