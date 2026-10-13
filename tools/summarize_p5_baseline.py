#!/usr/bin/env python3
"""Summarize P5 Middlebury regional quality baseline into Markdown table."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "p5_baseline" / "middlebury_f_mode_baseline.csv"

def summarize(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} scenes in {csv_path.name}\n")

    headers = [
        "Scene", "Time (ms)", "Peak MB", "Valid %", "All EPE", "All Bad2%",
        "Noc Bad2%", "Occ Bad2%", "Bnd Bad2%", "LowTex Bad2%", "HighTex Bad2%"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    tot_time = 0.0
    tot_all_pixels = 0
    tot_all_valid_pixels = 0
    tot_all_bad2_count = 0
    tot_all_epe_sum = 0.0

    tot_noc_pixels = 0
    tot_noc_bad2_count = 0

    tot_occ_pixels = 0
    tot_occ_bad2_count = 0

    tot_bnd_pixels = 0
    tot_bnd_bad2_count = 0

    tot_lowtex_pixels = 0
    tot_lowtex_bad2_count = 0

    tot_hightex_pixels = 0
    tot_hightex_bad2_count = 0

    max_peak_mb = 0.0

    for r in rows:
        case = r["case"]
        time_ms = float(r["time_total_ms"])
        tot_time += time_ms

        peak_mb = float(r.get("peak_workspace_bytes", 0)) / (1024 * 1024)
        if peak_mb == 0.0:
            peak_mb = float(r.get("peak_bytes", 0)) / (1024 * 1024)
        if peak_mb > max_peak_mb:
            max_peak_mb = peak_mb

        valid_ratio = float(r["all_valid_ratio"]) * 100.0
        all_epe = float(r["all_epe"])
        all_bad2 = float(r["all_bad20"])

        noc_bad2 = float(r["noc_bad20"])
        occ_bad2 = float(r["occ_bad20"])
        bnd_bad2 = float(r["bnd_bad20"])
        lowtex_bad2 = float(r["lowtex_bad20"])
        hightex_bad2 = float(r["hightex_bad20"])

        all_px = int(r["all_gt_pixels"])
        valid_px = int(float(r["all_valid_ratio"]) * all_px)
        tot_all_pixels += all_px
        tot_all_valid_pixels += valid_px
        tot_all_bad2_count += int(all_bad2 * valid_px / 100.0)
        tot_all_epe_sum += all_epe * valid_px

        noc_px = int(r["noc_pixels"])
        tot_noc_pixels += noc_px
        tot_noc_bad2_count += int(noc_bad2 * noc_px / 100.0)

        occ_px = int(r["occ_pixels"])
        tot_occ_pixels += occ_px
        tot_occ_bad2_count += int(occ_bad2 * occ_px / 100.0)

        bnd_px = int(r["bnd_pixels"])
        tot_bnd_pixels += bnd_px
        tot_bnd_bad2_count += int(bnd_bad2 * bnd_px / 100.0)

        lowtex_px = int(r["lowtex_pixels"])
        tot_lowtex_pixels += lowtex_px
        tot_lowtex_bad2_count += int(lowtex_bad2 * lowtex_px / 100.0)

        hightex_px = int(r["hightex_pixels"])
        tot_hightex_pixels += hightex_px
        tot_hightex_bad2_count += int(hightex_bad2 * hightex_px / 100.0)

        print(f"| {case:<12} | {time_ms:9.2f} | {peak_mb:7.1f} | {valid_ratio:6.2f}% | {all_epe:7.3f} | {all_bad2:7.2f}% | {noc_bad2:7.2f}% | {occ_bad2:7.2f}% | {bnd_bad2:7.2f}% | {lowtex_bad2:8.2f}% | {hightex_bad2:9.2f}% |")

    # Global aggregate
    agg_valid = (tot_all_valid_pixels / tot_all_pixels * 100.0) if tot_all_pixels > 0 else 0.0
    agg_epe = (tot_all_epe_sum / tot_all_valid_pixels) if tot_all_valid_pixels > 0 else 0.0
    agg_bad2 = (tot_all_bad2_count / tot_all_valid_pixels * 100.0) if tot_all_valid_pixels > 0 else 0.0
    agg_noc_bad2 = (tot_noc_bad2_count / tot_noc_pixels * 100.0) if tot_noc_pixels > 0 else 0.0
    agg_occ_bad2 = (tot_occ_bad2_count / tot_occ_pixels * 100.0) if tot_occ_pixels > 0 else 0.0
    agg_bnd_bad2 = (tot_bnd_bad2_count / tot_bnd_pixels * 100.0) if tot_bnd_pixels > 0 else 0.0
    agg_lowtex_bad2 = (tot_lowtex_bad2_count / tot_lowtex_pixels * 100.0) if tot_lowtex_pixels > 0 else 0.0
    agg_hightex_bad2 = (tot_hightex_bad2_count / tot_hightex_pixels * 100.0) if tot_hightex_pixels > 0 else 0.0

    print("| " + " | ".join(["---"] * len(headers)) + " |")
    print(f"| **Overall / Agg** | {tot_time:9.2f} | {max_peak_mb:7.1f} | {agg_valid:6.2f}% | {agg_epe:7.3f} | {agg_bad2:7.2f}% | {agg_noc_bad2:7.2f}% | {agg_occ_bad2:7.2f}% | {agg_bnd_bad2:7.2f}% | {agg_lowtex_bad2:8.2f}% | {agg_hightex_bad2:9.2f}% |")

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    summarize(path)
