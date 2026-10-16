#!/usr/bin/env python3
"""Bit-exact byte-level verification between explicit QualityV2 flags and native --preset quality_v2.

Compares raw exported .pfm disparity files byte-for-byte and via SHA-256 hashes across
all 15 Middlebury 2014 benchmark scenes.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
BIN = ROOT / "build" / ("codex-vs2022/Release/apg-benchmark.exe" if os.name == "nt" else "codex-linux-ninja/apg-benchmark")
RESULTS = ROOT / "results" / "p5_7_study"
DIR_EXPLICIT = RESULTS / "disp_explicit"
DIR_PRESET = RESULTS / "disp_preset"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_benchmark(save_dir: Path, preset: bool):
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dummy_csv = save_dir / "out.csv"
    cmd = [
        str(BIN),
        "--manifest", str(MANIFEST),
        "--backend", "packed",
        "--output", str(dummy_csv),
        "--warmup", "0",
        "--repeat", "1",
        "--save_disp", str(save_dir),
    ]

    if preset:
        cmd.extend(["--preset", "quality_v2"])
    else:
        cmd.extend([
            "--modes", "F",
            "--hierarchical",
            "--hierarchical-radius", "4",
            "--inpaint-method", "shadow_reclass",
            "--inpaint-gap", "48",
        ])

    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Benchmark run failed for preset={preset}")


def main():
    if not BIN.exists():
        print(f"Error: benchmark binary not found at {BIN}")
        sys.exit(1)

    print("=" * 80)
    print("P5.7 Preset Parity Verification: Explicit Flags vs Native --preset quality_v2")
    print(f"Binary: {BIN}")
    print(f"Manifest: {MANIFEST} (15 Middlebury scenes)")
    print("=" * 80)

    print("\n1. Running explicit QualityV2 configuration...")
    run_benchmark(DIR_EXPLICIT, preset=False)

    print("2. Running native --preset quality_v2 configuration...")
    run_benchmark(DIR_PRESET, preset=True)

    # Find all PFM files in DIR_EXPLICIT
    explicit_pfms = sorted(DIR_EXPLICIT.glob("*_disp.pfm"))
    if not explicit_pfms:
        print("Error: No .pfm files generated in explicit directory!")
        sys.exit(1)

    print(f"\n3. Comparing SHA-256 byte hashes of raw .pfm disparity maps ({len(explicit_pfms)} scenes)...\n")
    print(f"| {'Scene':<15} | {'Explicit SHA-256 (prefix)':<26} | {'Preset SHA-256 (prefix)':<26} | {'Parity':<10} |")
    print(f"|:{'-'*15}:|:{'-'*26}:|:{'-'*26}:|:{'-'*10}:|")

    all_matched = True
    scene_count = 0

    for exp_path in explicit_pfms:
        filename = exp_path.name
        preset_path = DIR_PRESET / filename
        scene_name = filename.replace("_F_disp.pfm", "").replace("_disp.pfm", "")

        if not preset_path.exists():
            print(f"| {scene_name:<15} | MISSING IN PRESET          | {'N/A':<26} | FAIL       |")
            all_matched = False
            continue

        exp_hash = compute_sha256(exp_path)
        preset_hash = compute_sha256(preset_path)

        match = (exp_hash == preset_hash)
        if not match:
            all_matched = False

        status = "MATCH (bit-exact)" if match else "MISMATCH"
        print(f"| {scene_name:<15} | {exp_hash[:24]}.. | {preset_hash[:24]}.. | {status:<10} |")
        scene_count += 1

    print("\n" + "=" * 80)
    if all_matched and scene_count == 15:
        print(f"VERDICT: PASS - 15/15 scenes bit-exact identical (100% byte-for-byte SHA-256 invariance)")
        print("=" * 80)
        sys.exit(0)
    else:
        print(f"VERDICT: FAIL - Disparity discrepancies detected (matched {scene_count}/15)")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
