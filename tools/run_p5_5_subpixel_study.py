#!/usr/bin/env python3
"""P5.5 Subpixel Refinement Study Runner and Summarizer.

Executes Candidates A, B, C, D across Lane L (Legacy Mode F) and Lane H (Hierarchical R=4).
Aggregates exact Middlebury metrics (EPE, Bad0.5, Bad1.0, Bad2_valid, Bad2_all, Valid%)
weighted by valid_est_pixels, as well as subpixel telemetry diagnostics.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
RESULTS_DIR = ROOT / "results" / "p5_5_study"
BENCHMARK_BIN = ROOT / "build" / (
    "codex-vs2022/Release/apg-benchmark.exe"
    if os.name == "nt"
    else "codex-linux-ninja/apg-benchmark"
)

REGIONS = [
    ("all", "All"),
    ("noc", "Non-Occ (Noc)"),
    ("occ", "Occluded (Occ)"),
    ("bnd", "Boundary (Bnd)"),
    ("nonbnd", "Non-Boundary"),
    ("lowtex", "Low Texture"),
    ("hightex", "High Texture"),
]

CANDIDATES = [
    ("A_legacy", "legacy", 0.0),
    ("B_curv_tau4", "curv_gate", 4.0),
    ("B_curv_tau2", "curv_gate", 2.0),
    ("B_curv_tau8", "curv_gate", 8.0),
    ("C_conf_tau005", "conf_gate", 0.05),
    ("C_conf_tau002", "conf_gate", 0.02),
    ("C_conf_tau010", "conf_gate", 0.10),
    ("D_stab_lam4", "stabilized", 4.0),
    ("D_stab_lam2", "stabilized", 2.0),
    ("D_stab_lam8", "stabilized", 8.0),
]


def run_benchmark_cmd(lane: str, cand_name: str, method: str, param: float, warmup: int = 1, repeat: int = 3) -> Path:
    out_csv = RESULTS_DIR / f"{lane}_{cand_name}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 500:
        print(f"[{lane}:{cand_name}] CSV exists: {out_csv.name}, skipping execution.")
        return out_csv

    cmd = [
        str(BENCHMARK_BIN),
        "--manifest", str(MANIFEST),
        "--modes", "F",
        "--backend", "packed",
        "--output", str(out_csv),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--subpixel-method", method,
        "--subpixel-param", str(param),
        "--subpixel-telem",
    ]

    if lane == "lane_h":
        cmd.extend(["--hierarchical", "--hierarchical-radius", "4"])

    print(f"[{lane}:{cand_name}] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Benchmark failed for {lane} {cand_name}")
    return out_csv


def extract_region_data(row: Dict[str, Any], reg_key: str) -> Dict[str, float]:
    tot_key = f"{reg_key}_gt_pixels" if f"{reg_key}_gt_pixels" in row else f"{reg_key}_pixels"
    tot_pixels = float(row.get(tot_key, 0.0))
    valid_est = float(row.get(f"{reg_key}_valid_est_pixels", 0.0))
    if valid_est == 0.0 and tot_pixels > 0.0:
        valid_est = float(row.get(f"{reg_key}_valid_ratio", 1.0)) * tot_pixels

    valid_ratio = (valid_est / tot_pixels) if tot_pixels > 0 else 1.0
    epe = float(row.get(f"{reg_key}_epe", 0.0))
    bad05 = float(row.get(f"{reg_key}_bad05", 0.0))
    bad10 = float(row.get(f"{reg_key}_bad10", 0.0))
    bad20 = float(row.get(f"{reg_key}_bad20", 0.0))
    bad30 = float(row.get(f"{reg_key}_bad30", 0.0))
    bad2_all = (1.0 - valid_ratio) * 100.0 + valid_ratio * bad20

    return {
        "total_pixels": tot_pixels,
        "valid_est_pixels": valid_est,
        "valid_ratio": valid_ratio,
        "epe": epe,
        "bad05": bad05,
        "bad10": bad10,
        "bad20": bad20,
        "bad30": bad30,
        "bad2_all": bad2_all,
    }


def analyze_csv(csv_path: Path) -> Dict[str, Any]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    tot_time_ms = 0.0
    reg_acc = {
        k: {
            "sum_tot": 0.0,
            "sum_valid": 0.0,
            "sum_epe_weighted": 0.0,
            "sum_bad05_weighted": 0.0,
            "sum_bad10_weighted": 0.0,
            "sum_bad20_weighted": 0.0,
            "sum_bad30_weighted": 0.0,
        }
        for k, _ in REGIONS
    }

    tot_candidate_px = 0
    tot_applied_px = 0
    tot_degen = 0
    tot_clamped = 0
    tot_neg_curv = 0
    sum_mean_delta = 0.0
    count_scenes = len(rows)

    for r in rows:
        tot_time_ms += float(r.get("time_total_ms", 0.0))
        tot_candidate_px += int(float(r.get("subpixel_candidate_px", 0)))
        tot_applied_px += int(float(r.get("subpixel_applied_px", 0)))
        tot_degen += int(float(r.get("subpixel_degenerate_count", 0)))
        tot_clamped += int(float(r.get("subpixel_clamped_count", 0)))
        tot_neg_curv += int(float(r.get("subpixel_neg_curv_count", 0)))
        sum_mean_delta += float(r.get("subpixel_mean_abs_delta", 0.0))

        for k, _ in REGIONS:
            rd = extract_region_data(r, k)
            v = rd["valid_est_pixels"]
            reg_acc[k]["sum_tot"] += rd["total_pixels"]
            reg_acc[k]["sum_valid"] += v
            reg_acc[k]["sum_epe_weighted"] += rd["epe"] * v
            reg_acc[k]["sum_bad05_weighted"] += rd["bad05"] * v
            reg_acc[k]["sum_bad10_weighted"] += rd["bad10"] * v
            reg_acc[k]["sum_bad20_weighted"] += rd["bad20"] * v
            reg_acc[k]["sum_bad30_weighted"] += rd["bad30"] * v

    # Final region metrics
    reg_metrics = {}
    for k, name in REGIONS:
        v = reg_acc[k]["sum_valid"]
        tot = reg_acc[k]["sum_tot"]
        vr = (v / tot) if tot > 0 else 1.0
        epe = (reg_acc[k]["sum_epe_weighted"] / v) if v > 0 else 0.0
        bad05 = (reg_acc[k]["sum_bad05_weighted"] / v) if v > 0 else 0.0
        bad10 = (reg_acc[k]["sum_bad10_weighted"] / v) if v > 0 else 0.0
        bad20 = (reg_acc[k]["sum_bad20_weighted"] / v) if v > 0 else 0.0
        bad30 = (reg_acc[k]["sum_bad30_weighted"] / v) if v > 0 else 0.0
        bad2_all = (1.0 - vr) * 100.0 + vr * bad20
        reg_metrics[k] = {
            "name": name,
            "total_pixels": tot,
            "valid_est_pixels": v,
            "valid_ratio_pct": vr * 100.0,
            "epe": epe,
            "bad05": bad05,
            "bad10": bad10,
            "bad20": bad20,
            "bad30": bad30,
            "bad2_all": bad2_all,
        }

    applied_ratio_pct = (tot_applied_px / tot_candidate_px * 100.0) if tot_candidate_px > 0 else 0.0
    clamped_ratio_pct = (tot_clamped / tot_applied_px * 100.0) if tot_applied_px > 0 else 0.0

    return {
        "time_ms": tot_time_ms,
        "regions": reg_metrics,
        "telemetry": {
            "candidate_px": tot_candidate_px,
            "applied_px": tot_applied_px,
            "applied_ratio_pct": applied_ratio_pct,
            "clamped_count": tot_clamped,
            "clamped_ratio_pct": clamped_ratio_pct,
            "degenerate_count": tot_degen,
            "negative_curvature_count": tot_neg_curv,
            "mean_abs_delta": (sum_mean_delta / count_scenes) if count_scenes > 0 else 0.0,
        }
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for lane in ["lane_l", "lane_h"]:
        results[lane] = {}
        for cand_name, method, param in CANDIDATES:
            csv_path = run_benchmark_cmd(lane, cand_name, method, param)
            results[lane][cand_name] = analyze_csv(csv_path)

    # Print summary tables
    for lane, lane_title in [("lane_l", "Lane L: Legacy Mode F (Full Range)"), ("lane_h", "Lane H: Hierarchical Mode F (R=4)")]:
        print(f"\n# {lane_title}\n")
        print("| Candidate | Param | Overall EPE | Bad0.5 (%) | Bad1.0 (%) | Bad2.0 (%) | Valid (%) | Bad2_all (%) | Time (ms) |")
        print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
        base_epe = results[lane]["A_legacy"]["regions"]["all"]["epe"]
        base_bad05 = results[lane]["A_legacy"]["regions"]["all"]["bad05"]
        base_bad10 = results[lane]["A_legacy"]["regions"]["all"]["bad10"]
        base_bad20 = results[lane]["A_legacy"]["regions"]["all"]["bad20"]
        base_vr = results[lane]["A_legacy"]["regions"]["all"]["valid_ratio_pct"]
        base_bad2all = results[lane]["A_legacy"]["regions"]["all"]["bad2_all"]
        base_time = results[lane]["A_legacy"]["time_ms"]

        for cand_name, method, param in CANDIDATES:
            m = results[lane][cand_name]["regions"]["all"]
            t = results[lane][cand_name]["time_ms"]
            d_epe = m["epe"] - base_epe
            d_bad05 = m["bad05"] - base_bad05
            d_bad10 = m["bad10"] - base_bad10
            d_bad20 = m["bad20"] - base_bad20
            d_vr = m["valid_ratio_pct"] - base_vr
            d_bad2all = m["bad2_all"] - base_bad2all
            d_time = (t - base_time) / base_time * 100.0

            cand_label = cand_name
            p_label = f"{param}" if param > 0 else "-"
            print(f"| {cand_label} | {p_label} | {m['epe']:.3f} ({d_epe:+.3f}) | {m['bad05']:.2f}% ({d_bad05:+.2f}%) | {m['bad10']:.2f}% ({d_bad10:+.2f}%) | {m['bad20']:.2f}% ({d_bad20:+.2f}%) | {m['valid_ratio_pct']:.2f}% ({d_vr:+.2f}%) | {m['bad2_all']:.2f}% ({d_bad2all:+.2f}%) | {t:.1f} ({d_time:+.1f}%) |")

        print(f"\n### Subpixel Diagnostics ({lane_title})\n")
        print("| Candidate | Candidate Px | Applied Px | Applied% | Clamped Px | Clamped% | Degenerate | NegCurv | Mean |d| |")
        print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
        for cand_name, method, param in CANDIDATES:
            telem = results[lane][cand_name]["telemetry"]
            print(f"| {cand_name} | {telem['candidate_px']:,} | {telem['applied_px']:,} | {telem['applied_ratio_pct']:.2f}% | {telem['clamped_count']:,} | {telem['clamped_ratio_pct']:.2f}% | {telem['degenerate_count']:,} | {telem['negative_curvature_count']:,} | {telem['mean_abs_delta']:.4f} |")


if __name__ == "__main__":
    main()
