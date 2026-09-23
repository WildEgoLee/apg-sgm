# APG-SGM Baseline & Ablation Benchmarking

This directory contains the benchmarking infrastructure for `apg-sgm`. The benchmark runner `apg-benchmark` evaluates stereo matching accuracy, robustness, and execution efficiency across standardized ablation configurations and datasets.

---

## 1. Single-Variable Ablation Matrix

The ablation matrix evaluates one single component at a time, moving sequentially from raw Census baseline to full high-quality APG-SGM:

| ID | Description | Cost | Cross Aggregation | SGM Penalty P2 | Prior Estimation | Refinement | SGM Paths |
|---|---|---|---|---|---|---|---|
| **A** | Raw Census Baseline | Census | Off | Fixed | Off | Off | 4 |
| **B** | Multi-Cost Fusion | Census + AD + Grad | Off | Fixed | Off | Off | 4 |
| **C** | Cross-Arm Aggregation | Census + AD + Grad | On (1 iter) | Fixed | Off | Off | 4 |
| **D** | Gradient-Adaptive P2 | Census + AD + Grad | On (1 iter) | Adaptive | Off | Off | 4 |
| **E** | APG Prior Disparity | Census + AD + Grad | On (1 iter) | Adaptive | On | Off | 4 |
| **F** | Propagative Refiner | Census + AD + Grad | On (1 iter) | Adaptive | On | On | 4 |
| **G** | High-Quality 8-Path | Census + AD + Grad | On (1 iter) | Adaptive | On | On | 8 |

---

## 2. Benchmark CLI Usage

`apg-benchmark` is built automatically with CMake:

```bash
# Run all ablation levels on a manifest:
apg-benchmark --manifest benchmarks/manifests/synthetic.txt --output results/ablation.csv

# Run specific ablations (e.g., A, E, G):
apg-benchmark --manifest benchmarks/manifests/synthetic.txt --ablation A,E,G --output results/ablation_subset.csv

# Configure warmup and repeats (median of repeats reported):
apg-benchmark --manifest benchmarks/manifests/synthetic.txt --warmup 2 --repeat 5

# Save disparity outputs (PFM and preview PGM):
apg-benchmark --manifest benchmarks/manifests/synthetic.txt --save_disp results/disp_maps
```

### CLI Arguments

- `--manifest <path>`: Path to manifest text file (required).
- `--ablation <id|all>`: Ablation experiment to run: `A`, `B`, `C`, `D`, `E`, `F`, `G`, comma-separated list, or `all` (default: `all`).
- `--output <path>`: Output CSV file destination (default: `results/ablation.csv`).
- `--warmup <N>`: Warmup iterations before timing (default: 2).
- `--repeat <N>`: Timed repeats per test case (default: 5, reports median).
- `--threads <N>`: OpenMP threads to use (default: 0 = system concurrency).
- `--save_disp <dir>`: Optional directory to save output disparity maps (`.pfm` float and `.pgm` visualization).

---

## 3. Dataset Manifest Format

Manifest files (`*.txt`) define test cases. Each line has the format:

```text
# case_name left_img right_img gt_disp [dmax] [dmin]
synth_blocks ../data/synthetic/synth_blocks_left.pgm ../data/synthetic/synth_blocks_right.pgm ../data/synthetic/synth_blocks_gt.pfm 32 0
```

- Relative paths in the manifest are resolved relative to the directory containing the manifest file.
- Supported input image formats: Netpbm PGM / PPM (`P5` / `P6`).
- Supported ground truth format: 1-channel float PFM (`Pf`, bottom-to-top scanlines).
- Blank lines and lines starting with `#` are ignored.

---

## 4. Evaluation Metrics

The output CSV reports standard Middlebury / KITTI stereo evaluation metrics:

- `epe`: End-point error (mean absolute error $|d_{est} - d_{gt}|$ on valid ground truth pixels).
- `bad_0_5`, `bad_1_0`, `bad_2_0`, `bad_3_0`: Percentage of valid pixels with absolute error exceeding 0.5, 1.0, 2.0, and 3.0 pixels.
- `valid_ratio`: Percentage of ground truth pixels where stereo matching produced valid disparity.
- `lr_fail_ratio`: Percentage of pixels rejected by left-right consistency check.
- `edge_epe` / `nonedge_epe`: Disparity error evaluated separately on depth boundaries vs flat regions.
- `range_gt_recall`: Fraction of ground truth disparities captured within the prior `SearchRange [dmin, dmax]`.
- `time_total_ms`: End-to-end stereo matching execution time (median of repeated runs, excluding I/O).
- `time_cost_ms`, `time_cross_ms`, `time_sgm_ms`, `time_prior_ms`, `time_refine_ms`: Per-stage timing breakdown.
- `refine_epe_delta`: Change in EPE after local disparity refinement ($EPE_{after} - EPE_{before}$).
- `refine_improved_pct`: Percentage of pixels whose error decreased after refinement.
