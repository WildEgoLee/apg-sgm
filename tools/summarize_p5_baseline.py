#!/usr/bin/env python3
"""Summarize P5 Middlebury regional quality baseline into Markdown table.

Implements denominator-weighted regional aggregation based on valid_est_pixels
and anti-invalidation tri-gate metrics (Bad2_valid, Valid%, Bad2_all).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "p5_baseline" / "middlebury_f_mode_baseline.csv"

REGIONS = [
    ("all", "all"),
    ("noc", "noc"),
    ("occ", "occ"),
    ("bnd", "bnd"),
    ("lowtex", "lowtex"),
    ("hightex", "hightex"),
]


def extract_region_data(row: Dict[str, Any], reg_key: str) -> Dict[str, float]:
    """Extracts total_pixels, valid_est_pixels, epe, bad2, and computes derived bad2_all."""
    # Column naming differences: 'all_gt_pixels' vs 'noc_pixels' etc.
    if f"{reg_key}_gt_pixels" in row:
        tot_pixels = float(row[f"{reg_key}_gt_pixels"])
    elif f"{reg_key}_pixels" in row:
        tot_pixels = float(row[f"{reg_key}_pixels"])
    else:
        tot_pixels = 0.0

    # Prefer exact valid_est_pixels exported by benchmark; fallback to valid_ratio * tot_pixels
    if f"{reg_key}_valid_est_pixels" in row and row[f"{reg_key}_valid_est_pixels"] != "":
        valid_est = float(row[f"{reg_key}_valid_est_pixels"])
    else:
        vr = float(row.get(f"{reg_key}_valid_ratio", 1.0))
        valid_est = vr * tot_pixels

    valid_ratio = (valid_est / tot_pixels) if tot_pixels > 0 else 1.0
    epe = float(row.get(f"{reg_key}_epe", 0.0))
    bad2_valid = float(row.get(f"{reg_key}_bad20", 0.0))

    # Bad2_all: invalid pixels count as errors
    # Bad2_all% = (1 - valid_ratio) * 100 + valid_ratio * Bad2_valid%
    bad2_all = (1.0 - valid_ratio) * 100.0 + valid_ratio * bad2_valid

    return {
        "total_pixels": tot_pixels,
        "valid_est_pixels": valid_est,
        "valid_ratio": valid_ratio,
        "epe": epe,
        "bad2_valid": bad2_valid,
        "bad2_all": bad2_all,
    }


def aggregate_regions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mathematically rigorous cross-scene aggregations weighted by valid_est_pixels."""
    scene_data = []
    tot_time_ms = 0.0
    max_peak_mb = 0.0

    # Accumulators per region
    acc = {
        k: {
            "sum_tot_pixels": 0.0,
            "sum_valid_est": 0.0,
            "sum_bad2_weighted": 0.0,
            "sum_epe_weighted": 0.0,
        }
        for k, _ in REGIONS
    }

    for r in rows:
        case = r.get("case", "unknown")
        time_ms = float(r.get("time_total_ms", 0.0))
        tot_time_ms += time_ms

        peak_mb = float(r.get("peak_workspace_bytes", 0)) / (1024 * 1024)
        if peak_mb == 0.0:
            peak_mb = float(r.get("peak_bytes", 0)) / (1024 * 1024)
        if peak_mb > max_peak_mb:
            max_peak_mb = peak_mb

        cur_scene: Dict[str, Any] = {
            "case": case,
            "time_ms": time_ms,
            "peak_mb": peak_mb,
        }

        for reg_key, _ in REGIONS:
            rd = extract_region_data(r, reg_key)
            cur_scene[reg_key] = rd

            # Accumulate
            acc[reg_key]["sum_tot_pixels"] += rd["total_pixels"]
            acc[reg_key]["sum_valid_est"] += rd["valid_est_pixels"]
            acc[reg_key]["sum_bad2_weighted"] += rd["bad2_valid"] * rd["valid_est_pixels"]
            acc[reg_key]["sum_epe_weighted"] += rd["epe"] * rd["valid_est_pixels"]

        scene_data.append(cur_scene)

    # Compute global aggregated metrics
    agg: Dict[str, Any] = {
        "time_ms": tot_time_ms,
        "peak_mb": max_peak_mb,
    }

    for reg_key, _ in REGIONS:
        a = acc[reg_key]
        sum_valid = a["sum_valid_est"]
        sum_tot = a["sum_tot_pixels"]

        agg_valid_ratio = (sum_valid / sum_tot) if sum_tot > 0 else 0.0
        agg_epe = (a["sum_epe_weighted"] / sum_valid) if sum_valid > 0 else 0.0
        agg_bad2_valid = (a["sum_bad2_weighted"] / sum_valid) if sum_valid > 0 else 0.0
        agg_bad2_all = (1.0 - agg_valid_ratio) * 100.0 + agg_valid_ratio * agg_bad2_valid

        agg[reg_key] = {
            "total_pixels": sum_tot,
            "valid_est_pixels": sum_valid,
            "valid_ratio": agg_valid_ratio,
            "epe": agg_epe,
            "bad2_valid": agg_bad2_valid,
            "bad2_all": agg_bad2_all,
        }

    return {"scenes": scene_data, "aggregate": agg}


def summarize(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} scenes in {csv_path.name}\n")
    res = aggregate_regions(rows)
    scenes = res["scenes"]
    agg = res["aggregate"]

    headers = [
        "Scene", "Time (ms)", "Peak MB", "Valid %", "All EPE",
        "All Bad2%", "All Bad2_all%", "Noc Bad2%", "Occ Bad2%",
        "Bnd Bad2%", "LowTex Bad2%", "HighTex Bad2%"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in scenes:
        c = s["case"]
        t = s["time_ms"]
        pmb = s["peak_mb"]
        all_vr = s["all"]["valid_ratio"] * 100.0
        all_epe = s["all"]["epe"]
        all_b2 = s["all"]["bad2_valid"]
        all_b2_all = s["all"]["bad2_all"]
        noc_b2 = s["noc"]["bad2_valid"]
        occ_b2 = s["occ"]["bad2_valid"]
        bnd_b2 = s["bnd"]["bad2_valid"]
        low_b2 = s["lowtex"]["bad2_valid"]
        high_b2 = s["hightex"]["bad2_valid"]

        print(
            f"| {c:<12} | {t:9.2f} | {pmb:7.1f} | {all_vr:6.2f}% | {all_epe:7.3f} | "
            f"{all_b2:7.2f}% | {all_b2_all:11.2f}% | {noc_b2:7.2f}% | {occ_b2:7.2f}% | "
            f"{bnd_b2:7.2f}% | {low_b2:8.2f}% | {high_b2:9.2f}% |"
        )

    # Print global aggregate row
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    c_agg = "**Overall / Agg**"
    t_agg = agg["time_ms"]
    pmb_agg = agg["peak_mb"]
    all_vr_a = agg["all"]["valid_ratio"] * 100.0
    all_epe_a = agg["all"]["epe"]
    all_b2_a = agg["all"]["bad2_valid"]
    all_b2_all_a = agg["all"]["bad2_all"]
    noc_b2_a = agg["noc"]["bad2_valid"]
    occ_b2_a = agg["occ"]["bad2_valid"]
    bnd_b2_a = agg["bnd"]["bad2_valid"]
    low_b2_a = agg["lowtex"]["bad2_valid"]
    high_b2_a = agg["hightex"]["bad2_valid"]

    print(
        f"| {c_agg:<12} | {t_agg:9.2f} | {pmb_agg:7.1f} | {all_vr_a:6.2f}% | {all_epe_a:7.3f} | "
        f"{all_b2_a:7.2f}% | {all_b2_all_a:11.2f}% | {noc_b2_a:7.2f}% | {occ_b2_a:7.2f}% | "
        f"{bnd_b2_a:7.2f}% | {low_b2_a:8.2f}% | {high_b2_a:9.2f}% |"
    )

    print("\n### Anti-invalidation Tri-Gate Summary (All & Regional)")
    print("| Region | Bad2 (Valid) % | Valid Ratio % | Bad2_all (Unconstrained) % | Total Eval Px |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for reg_key, label in [("all", "All"), ("noc", "Non-Occluded (Noc)"), ("occ", "Occluded (Occ)"),
                           ("bnd", "Disparity Boundary (Bnd)"), ("lowtex", "Low Texture (LowTex)"),
                           ("hightex", "High Texture (HighTex)")]:
        r_info = agg[reg_key]
        r_b2 = r_info["bad2_valid"]
        r_vr = r_info["valid_ratio"] * 100.0
        r_b2_all = r_info["bad2_all"]
        r_tot = int(r_info["total_pixels"])
        print(f"| {label:<24} | {r_b2:12.2f}% | {r_vr:11.2f}% | {r_b2_all:24.2f}% | {r_tot:13,d} |")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    summarize(path)
