import sys
from pathlib import Path
import pytest

# Add repo root to sys.path to import tools.summarize_p5_baseline
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.summarize_p5_baseline import aggregate_regions, extract_region_data


def test_denominator_weighted_aggregation():
    """Verify that regional aggregation is weighted by valid_est_pixels, not total_pixels.

    Synthetic case from P5.0a audit:
    Scene A: 100 pixels, 100 valid, 10% Bad2  -> 10 error pixels
    Scene B: 100 pixels,  50 valid, 20% Bad2  -> 10 error pixels

    Correct aggregate Bad2: (10 + 10) / 150 = 13.333...%
    Naive unweighted average would be: (10 + 20) / 200 = 15.0% (INCORRECT)
    """
    rows = [
        {
            "case": "scene_A",
            "time_total_ms": "10.0",
            "peak_workspace_bytes": "1000000",
            "all_gt_pixels": "100",
            "all_valid_est_pixels": "100",
            "all_valid_ratio": "1.0",
            "all_epe": "1.0",
            "all_bad20": "10.0",
            "noc_pixels": "100",
            "noc_valid_est_pixels": "100",
            "noc_valid_ratio": "1.0",
            "noc_epe": "1.0",
            "noc_bad20": "10.0",
            "occ_pixels": "100",
            "occ_valid_est_pixels": "100",
            "occ_valid_ratio": "1.0",
            "occ_epe": "1.0",
            "occ_bad20": "10.0",
            "bnd_pixels": "100",
            "bnd_valid_est_pixels": "100",
            "bnd_valid_ratio": "1.0",
            "bnd_epe": "1.0",
            "bnd_bad20": "10.0",
            "lowtex_pixels": "100",
            "lowtex_valid_est_pixels": "100",
            "lowtex_valid_ratio": "1.0",
            "lowtex_epe": "1.0",
            "lowtex_bad20": "10.0",
            "hightex_pixels": "100",
            "hightex_valid_est_pixels": "100",
            "hightex_valid_ratio": "1.0",
            "hightex_epe": "1.0",
            "hightex_bad20": "10.0",
        },
        {
            "case": "scene_B",
            "time_total_ms": "10.0",
            "peak_workspace_bytes": "1000000",
            "all_gt_pixels": "100",
            "all_valid_est_pixels": "50",
            "all_valid_ratio": "0.5",
            "all_epe": "2.0",
            "all_bad20": "20.0",
            "noc_pixels": "100",
            "noc_valid_est_pixels": "50",
            "noc_valid_ratio": "0.5",
            "noc_epe": "2.0",
            "noc_bad20": "20.0",
            "occ_pixels": "100",
            "occ_valid_est_pixels": "50",
            "occ_valid_ratio": "0.5",
            "occ_epe": "2.0",
            "occ_bad20": "20.0",
            "bnd_pixels": "100",
            "bnd_valid_est_pixels": "50",
            "bnd_valid_ratio": "0.5",
            "bnd_epe": "2.0",
            "bnd_bad20": "20.0",
            "lowtex_pixels": "100",
            "lowtex_valid_est_pixels": "50",
            "lowtex_valid_ratio": "0.5",
            "lowtex_epe": "2.0",
            "lowtex_bad20": "20.0",
            "hightex_pixels": "100",
            "hightex_valid_est_pixels": "50",
            "hightex_valid_ratio": "0.5",
            "hightex_epe": "2.0",
            "hightex_bad20": "20.0",
        },
    ]

    res = aggregate_regions(rows)
    agg = res["aggregate"]

    # Verify All region
    expected_bad2 = (10.0 * 100 + 20.0 * 50) / 150.0  # = 2000 / 150 = 13.33333333%
    assert abs(agg["all"]["bad2_valid"] - expected_bad2) < 1e-6
    assert abs(agg["all"]["bad2_valid"] - 15.0) > 1.0  # Must NOT be 15.0%

    # Verify Occ region
    assert abs(agg["occ"]["bad2_valid"] - expected_bad2) < 1e-6

    # Verify Valid Ratio: 150 / 200 = 75.0%
    assert abs(agg["all"]["valid_ratio"] - 0.75) < 1e-6

    # Verify Derived Bad2_all: (1 - 0.75) * 100 + 0.75 * 13.33333333 = 25.0 + 10.0 = 35.0%
    assert abs(agg["all"]["bad2_all"] - 35.0) < 1e-6


def test_anti_invalidation_gaming_detection():
    """Verify that aggressive invalidation increases Bad2_all and reduces Valid%."""
    # Baseline scene: 100 pixels, all valid, 10% bad2
    row_baseline = {
        "case": "clean",
        "all_gt_pixels": "100",
        "all_valid_est_pixels": "100",
        "all_bad20": "10.0",
    }
    stats_base = extract_region_data(row_baseline, "all")
    assert stats_base["valid_ratio"] == 1.0
    assert stats_base["bad2_valid"] == 10.0
    assert stats_base["bad2_all"] == 10.0

    # Gaming candidate: marks 50 pixels as invalid to artificially drop Bad2_valid to 2%
    row_gaming = {
        "case": "gamed",
        "all_gt_pixels": "100",
        "all_valid_est_pixels": "50",
        "all_bad20": "2.0",
    }
    stats_gamed = extract_region_data(row_gaming, "all")
    assert stats_gamed["valid_ratio"] == 0.5
    assert stats_gamed["bad2_valid"] == 2.0  # Looks "improved" if only looking at Bad2_valid
    # But Bad2_all exposes the penalty: (1 - 0.5)*100 + 0.5*2.0 = 51.0%
    assert stats_gamed["bad2_all"] == 51.0
    assert stats_gamed["bad2_all"] > stats_base["bad2_all"]
