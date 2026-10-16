#!/usr/bin/env python3
"""P5.6 Occlusion / Mismatch Refinement Study Runner and Summarizer.

Executes Candidates across Lane L (Legacy Mode F) and Lane H (Hierarchical R=4).
Aggregates exact Middlebury metrics (EPE, Bad0.5, Bad1.0, Bad2_valid, Bad2_all, Valid%)
weighted by valid_est_pixels, inpainting attribution telemetry, and evaluates all
decision and safety gates under the P5.6 contract.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "middlebury2014.txt"
RESULTS_DIR = ROOT / "results" / "p5_6_study"
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

CANDIDATES: List[Tuple[str, str, int]] = [
    ("A_legacy_gap32", "legacy", 32),
    ("B_order_gap32", "disparity_order", 32),
    ("B_order_gap48", "disparity_order", 48),
    ("B_order_gap64", "disparity_order", 64),
    ("C_shadow_gap32", "shadow_reclass", 32),
    ("C_shadow_gap48", "shadow_reclass", 48),
    ("C_shadow_gap64", "shadow_reclass", 64),
]


def run_benchmark_cmd(
    lane: str,
    cand_name: str,
    method: str,
    gap: int,
    warmup: int = 1,
    repeat: int = 3,
    force: bool = False,
) -> Path:
    out_csv = RESULTS_DIR / f"{lane}_{cand_name}.csv"
    if not force and out_csv.exists() and out_csv.stat().st_size > 500:
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
        "--inpaint-method", method,
        "--inpaint-gap", str(gap),
        "--inpaint-telem",
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

    per_scene_occ_bad2 = {}
    tot_holes = 0
    tot_filled = 0
    tot_occ_two = 0
    tot_occ_left = 0
    tot_occ_right = 0
    tot_occ_unfilled = 0
    tot_mism_two = 0
    tot_mism_left = 0
    tot_mism_right = 0
    tot_mism_unfilled = 0
    tot_reclass_shadow = 0
    tot_hit_max_gap = 0

    for r in rows:
        scene = r.get("case", r.get("case_name", r.get("dataset_case", "")))
        tot_time_ms += float(r.get("time_total_ms", 0.0))
        tot_holes += int(float(r.get("inpaint_total_holes", 0)))
        tot_filled += int(float(r.get("inpaint_filled_holes", 0)))
        tot_occ_two += int(float(r.get("inpaint_occ_two_sided", 0)))
        tot_occ_left += int(float(r.get("inpaint_occ_left_only", 0)))
        tot_occ_right += int(float(r.get("inpaint_occ_right_only", 0)))
        tot_occ_unfilled += int(float(r.get("inpaint_occ_unfilled", 0)))
        tot_mism_two += int(float(r.get("inpaint_mism_two_sided", 0)))
        tot_mism_left += int(float(r.get("inpaint_mism_left_only", 0)))
        tot_mism_right += int(float(r.get("inpaint_mism_right_only", 0)))
        tot_mism_unfilled += int(float(r.get("inpaint_mism_unfilled", 0)))
        tot_reclass_shadow += int(float(r.get("inpaint_reclassified_shadow", 0)))
        tot_hit_max_gap += int(float(r.get("inpaint_hit_max_gap", 0)))

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

        per_scene_occ_bad2[scene] = float(r.get("occ_bad20", 0.0))

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

    fill_ratio_pct = (tot_filled / tot_holes * 100.0) if tot_holes > 0 else 0.0
    hit_gap_ratio_pct = (tot_hit_max_gap / tot_holes * 100.0) if tot_holes > 0 else 0.0

    return {
        "time_ms": tot_time_ms,
        "regions": reg_metrics,
        "per_scene_occ_bad2": per_scene_occ_bad2,
        "telemetry": {
            "total_holes": tot_holes,
            "filled_holes": tot_filled,
            "fill_ratio_pct": fill_ratio_pct,
            "occ_two_sided": tot_occ_two,
            "occ_left_only": tot_occ_left,
            "occ_right_only": tot_occ_right,
            "occ_unfilled": tot_occ_unfilled,
            "mism_two_sided": tot_mism_two,
            "mism_left_only": tot_mism_left,
            "mism_right_only": tot_mism_right,
            "mism_unfilled": tot_mism_unfilled,
            "reclass_shadow": tot_reclass_shadow,
            "hit_max_gap": tot_hit_max_gap,
            "hit_gap_ratio_pct": hit_gap_ratio_pct,
        },
    }


def main():
    force = "--force" in sys.argv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for lane in ["lane_l", "lane_h"]:
        results[lane] = {}
        for cand_name, method, gap in CANDIDATES:
            csv_path = run_benchmark_cmd(lane, cand_name, method, gap, force=force)
            results[lane][cand_name] = analyze_csv(csv_path)

    # Print summary tables
    for lane, lane_title in [
        ("lane_l", "Lane L: Legacy Mode F (Canonical Full Range)"),
        ("lane_h", "Lane H: Hierarchical Mode F (R=4)"),
    ]:
        print(f"\n================================================================================")
        print(f"# {lane_title}")
        print(f"================================================================================\n")

        base_res = results[lane]["A_legacy_gap32"]
        base_occ_bad2 = base_res["regions"]["occ"]["bad20"]
        base_all_epe = base_res["regions"]["all"]["epe"]
        base_bnd_bad2 = base_res["regions"]["bnd"]["bad20"]
        base_occ_valid = base_res["regions"]["occ"]["valid_ratio_pct"]
        base_occ_bad2all = base_res["regions"]["occ"]["bad2_all"]
        base_time = base_res["time_ms"]
        base_scenes = base_res["per_scene_occ_bad2"]

        print("### Candidate Quality & Gate Evaluation Matrix\n")
        print("| Candidate | Occ Bad2.0 | Rel. Occ Impv | All EPE (Rel%) | Bnd Bad2 (d_pp) | Occ Valid% (d_pp) | Occ Bad2_all | Non-Regr Scenes | Overhead | Verdict |")
        print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|")

        for cand_name, method, gap in CANDIDATES:
            cand_res = results[lane][cand_name]
            occ_bad2 = cand_res["regions"]["occ"]["bad20"]
            all_epe = cand_res["regions"]["all"]["epe"]
            bnd_bad2 = cand_res["regions"]["bnd"]["bad20"]
            occ_valid = cand_res["regions"]["occ"]["valid_ratio_pct"]
            occ_bad2all = cand_res["regions"]["occ"]["bad2_all"]
            t = cand_res["time_ms"]

            rel_occ_impv = (base_occ_bad2 - occ_bad2) / base_occ_bad2 * 100.0
            rel_epe_reg = (all_epe - base_all_epe) / base_all_epe * 100.0
            d_bnd_bad2 = bnd_bad2 - base_bnd_bad2
            d_occ_valid = occ_valid - base_occ_valid
            d_occ_bad2all = occ_bad2all - base_occ_bad2all
            overhead = (t - base_time) / base_time * 100.0

            cand_scenes = cand_res["per_scene_occ_bad2"]
            non_regr_count = sum(
                1 for s, val in cand_scenes.items() if val <= base_scenes.get(s, val) + 1e-3
            )

            g1_primary = rel_occ_impv >= 5.0
            g2_epe = rel_epe_reg <= 1.0
            g3_bnd = d_bnd_bad2 <= 1.0
            g4_valid = d_occ_valid >= -0.5
            g5_bad2all = d_occ_bad2all <= 0.05
            g6_overhead = overhead <= 5.0
            g7_scenes = non_regr_count >= 12

            if cand_name == "A_legacy_gap32":
                verdict = "BASELINE"
            elif g1_primary and g2_epe and g3_bnd and g4_valid and g5_bad2all and g6_overhead and g7_scenes:
                verdict = "**PASS (GO)**"
            else:
                verdict = "FAIL (NO-GO)"

            print(
                f"| {cand_name} | {occ_bad2:.2f}% | {rel_occ_impv:+.2f}% | {all_epe:.3f} ({rel_epe_reg:+.2f}%) | "
                f"{bnd_bad2:.2f}% ({d_bnd_bad2:+.2f}pp) | {occ_valid:.2f}% ({d_occ_valid:+.2f}pp) | "
                f"{occ_bad2all:.2f}% ({d_occ_bad2all:+.2f}pp) | {non_regr_count}/{len(base_scenes)} | "
                f"{overhead:+.1f}% | {verdict} |"
            )

        print(f"\n### Inpainting Route & Attribution Telemetry\n")
        print("| Candidate | Total Holes | Filled% | Occ Both | Occ L-only | Occ R-only | Mism Lerp | Reclass Shadow | Hit Max Gap |")
        print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
        for cand_name, method, gap in CANDIDATES:
            telem = results[lane][cand_name]["telemetry"]
            print(
                f"| {cand_name} | {telem['total_holes']:,} | {telem['fill_ratio_pct']:.2f}% | "
                f"{telem['occ_two_sided']:,} | {telem['occ_left_only']:,} | {telem['occ_right_only']:,} | "
                f"{telem['mism_two_sided']:,} | {telem['reclass_shadow']:,} | "
                f"{telem['hit_max_gap']:,} ({telem['hit_gap_ratio_pct']:.1f}%) |"
            )

        print(f"\n### Full Regional Accuracy Breakdown\n")
        print("| Candidate | All EPE | All Bad2.0 | Noc Bad2.0 | Occ Bad2.0 | Bnd Bad2.0 | Non-Bnd Bad2.0 | LowTex Bad2.0 | Valid% |")
        print("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
        for cand_name, method, gap in CANDIDATES:
            r = results[lane][cand_name]["regions"]
            print(
                f"| {cand_name} | {r['all']['epe']:.3f} | {r['all']['bad20']:.2f}% | "
                f"{r['noc']['bad20']:.2f}% | {r['occ']['bad20']:.2f}% | {r['bnd']['bad20']:.2f}% | "
                f"{r['nonbnd']['bad20']:.2f}% | {r['lowtex']['bad20']:.2f}% | {r['all']['valid_ratio_pct']:.2f}% |"
            )


if __name__ == "__main__":
    main()
