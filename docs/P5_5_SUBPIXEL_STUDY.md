# Phase 5.5: Subpixel Refinement Study

**Status**: **FEASIBILITY STUDY COMPLETE — ALGORITHM NO-GO; IMPLEMENTATION UNIFICATION GO**  
**Date**: October 2026  
**Scope**: WTA integer disparity $\to$ local cost-curve subpixel estimator $\to$ float disparity  
**Dataset**: Middlebury 2014 Stereo Benchmark (15 full scenes, quarter resolution, scale=1.0)  
**Evaluation Protocol**: Interleaved matched runs, exact exported `valid_est_pixels` denominator-weighted aggregation, anti-invalidation tri-gate ($\text{Bad2}_{valid}$, $\text{Valid}\%$, $\text{Bad2}_{all}$).

---

## 1. Executive Summary & Feasibility Verdict

Phase 5.5 investigated whether altering the post-WTA local cost-curve subpixel estimator beyond the legacy 3-point parabola yields quantifiable, safe accuracy benefits on Middlebury benchmarks without regional or performance regressions.

### Conclusion

> **Algorithm Verdict: NO-GO**  
> Within the studied three-point local estimator family, current pipeline, and Middlebury 2014 evaluation, no Pareto-improving replacement was found. The production pipeline retains the legacy three-point quadratic estimator.

> **Implementation Verdict: GO**  
> Consolidates 5 historically duplicated quadratic subpixel implementations (`sgm_optimizer.cpp` dense/packed, `sgm_optimizer_avx2.cpp`, `pipeline.cpp` dense/packed right WTA, and `hierarchical_stereo.cpp`) into a single inlined, zero-overhead `SubpixelEstimator::compute_legacy_delta` with bit-exact preservation of legacy outputs and zero runtime regression.

---

## 2. Theoretical Formulations & Studied Candidates

Given three discrete aggregated SGM costs at integer disparities $d-1$, $d$, and $d+1$:
$$C_m = C(x, y, d-1), \quad C_0 = C(x, y, d), \quad C_p = C(x, y, d+1)$$
where $d$ is the discrete WTA winner ($C_0 \le C_m$ and $C_0 \le C_p$).

### 1. Candidate A: Legacy Quadratic Fit (Production Baseline)
$$\Delta_{num} = C_m - C_p, \quad \Delta_{curv} = C_m - 2C_0 + C_p$$
$$\delta = \text{clamp}\left( \frac{1}{2} \cdot \frac{\Delta_{num}}{\Delta_{curv}}, -0.5, +0.5 \right) \quad \text{if } |\Delta_{curv}| > 10^{-6}, \text{ else } 0$$

### 2. Candidate B: Curvature-Gated Quadratic Fit
Guards against non-convex basins and flat curvature by requiring strict convexity and a minimum curvature $\tau_{\text{curv}} \in \{2.0, 4.0, 8.0\}$:
$$\delta = \begin{cases}
\text{clamp}\left( \frac{1}{2} \cdot \frac{C_m - C_p}{C_m - 2C_0 + C_p}, -0.5, +0.5 \right), & \text{if } C_0 < C_m \land C_0 < C_p \land \Delta_{curv} \ge \tau_{\text{curv}} \\
0, & \text{otherwise (fallback to integer } d\text{)}
\end{cases}$$

### 3. Candidate C: Confidence/Uniqueness-Gated Quadratic Fit
Restricts subpixel shifts to pixels meeting a uniqueness threshold ($u = \frac{C_{second} - C_{best}}{\max(C_{best}, 1)} \ge \tau_{\text{uniq}} \in \{0.02, 0.05, 0.10\}$), falling back to integer disparity on ambiguous matches:
$$\delta = \begin{cases}
\delta_{\text{quadratic}}, & \text{if } u \ge \tau_{\text{uniq}} \\
0, & \text{otherwise}
\end{cases}$$

### 4. Candidate D: Stabilized Regularized Quadratic Fit
Adds Tikhonov-style regularization $\lambda \in \{2.0, 4.0, 8.0\}$ to smoothly dampen subpixel shifts in shallow basins:
$$\delta = \begin{cases}
\text{clamp}\left( \frac{1}{2} \cdot \frac{C_m - C_p}{\Delta_{curv} + \lambda}, -0.5, +0.5 \right), & \text{if } C_0 \le C_m \land C_0 \le C_p \land \Delta_{curv} > 0 \\
0, & \text{otherwise}
\end{cases}$$

---

## 3. Experimental Evaluation Matrix

Evaluation was executed across all 15 scenes of Middlebury 2014 (`Adirondack`, `ArtL`, `Jadeplant`, `Motorcycle`, `MotorcycleE`, `Piano`, `PianoL`, `Pipes`, `Playroom`, `Playtable`, `PlaytableP`, `Recycle`, `Shelves`, `Teddy`, `Vintage`).  
All aggregations are weighted by exact exported `valid_est_pixels`.

### Lane L: Legacy Mode F (Full Search Range)

| Candidate | Param | Overall EPE (px) | Bad0.5 (%) | Bad1.0 (%) | Bad2.0 (%) | Valid (%) | Bad2_all (%) | Time (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Legacy)** | Baseline | **3.477** (+0.000) | **41.75%** (+0.00%) | **29.55%** (+0.00%) | **22.41%** (+0.00%) | **99.73%** | **22.62%** | **1428.1** |
| B (CurvGate) | $\tau=2.0$ | 3.477 (-0.000) | 41.75% (+0.01%) | 29.56% (+0.01%) | 22.42% (+0.01%) | 99.73% | 22.63% | 1415.0 |
| B (CurvGate) | $\tau=4.0$ | 3.479 (+0.002) | 41.75% (+0.01%) | 29.55% (+0.00%) | 22.43% (+0.02%) | 99.73% | 22.63% | 1422.7 |
| B (CurvGate) | $\tau=8.0$ | 3.477 (+0.000) | 41.78% (+0.04%) | 29.57% (+0.02%) | 22.41% (+0.00%) | 99.73% | 22.62% | 1415.7 |
| C (ConfGate) | $\tau=0.02$ | 3.476 (-0.001) | 41.73% (-0.01%) | 29.52% (-0.03%) | 22.36% (-0.04%) | 99.73% | 22.57% | 1414.2 |
| C (ConfGate) | $\tau=0.05$ | 3.478 (+0.002) | 41.74% (-0.01%) | 29.57% (+0.02%) | 22.42% (+0.01%) | 99.73% | 22.62% | 1416.4 |
| C (ConfGate) | $\tau=0.10$ | 3.474 (-0.003) | 41.75% (+0.01%) | 29.55% (-0.01%) | 22.41% (-0.00%) | 99.73% | 22.61% | 1413.3 |
| D (Stabilized)| $\lambda=2.0$ | 3.477 (+0.001) | 41.75% (+0.01%) | 29.56% (+0.01%) | 22.40% (-0.01%) | 99.73% | 22.61% | 1419.4 |
| D (Stabilized)| $\lambda=4.0$ | 3.485 (+0.008) | 41.79% (+0.05%) | 29.59% (+0.04%) | 22.45% (+0.04%) | 99.73% | 22.66% | 1415.0 |
| D (Stabilized)| $\lambda=8.0$ | 3.473 (-0.004) | 41.79% (+0.05%) | 29.52% (-0.03%) | 22.39% (-0.02%) | 99.73% | 22.60% | 1415.3 |

### Lane H: Hierarchical Mode F ($S=2, R=4$)

| Candidate | Param | Overall EPE (px) | Bad0.5 (%) | Bad1.0 (%) | Bad2.0 (%) | Valid (%) | Bad2_all (%) | Time (ms) | Speedup vs Lane L |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Legacy)** | Baseline | **3.390** (+0.000) | **41.57%** (+0.00%) | **29.59%** (+0.00%) | **22.31%** (+0.00%) | **99.83%** | **22.44%** | **1242.1** | **1.150×** |
| B (CurvGate) | $\tau=2.0$ | 3.391 (+0.001) | 41.57% (-0.00%) | 29.55% (-0.04%) | 22.28% (-0.03%) | 99.83% | 22.40% | 1261.1 | 1.132× |
| B (CurvGate) | $\tau=4.0$ | 3.390 (+0.000) | 41.61% (+0.04%) | 29.59% (-0.00%) | 22.30% (-0.01%) | 99.83% | 22.42% | 1261.9 | 1.132× |
| B (CurvGate) | $\tau=8.0$ | 3.389 (-0.001) | 41.60% (+0.02%) | 29.60% (+0.00%) | 22.29% (-0.02%) | 99.83% | 22.41% | 1233.0 | 1.158× |
| C (ConfGate) | $\tau=0.02$ | 3.389 (-0.001) | 41.59% (+0.01%) | 29.57% (-0.02%) | 22.25% (-0.06%) | 99.83% | 22.38% | 1225.7 | 1.165× |
| C (ConfGate) | $\tau=0.05$ | 3.398 (+0.008) | 41.61% (+0.03%) | 29.58% (-0.01%) | 22.30% (-0.01%) | 99.83% | 22.43% | 1210.3 | 1.180× |
| C (ConfGate) | $\tau=0.10$ | 3.394 (+0.004) | 41.61% (+0.04%) | 29.60% (+0.01%) | 22.30% (-0.01%) | 99.84% | 22.43% | 1222.5 | 1.168× |
| D (Stabilized)| $\lambda=2.0$ | 3.394 (+0.004) | 41.59% (+0.02%) | 29.60% (+0.00%) | 22.29% (-0.02%) | 99.83% | 22.42% | 1208.6 | 1.182× |
| D (Stabilized)| $\lambda=4.0$ | 3.395 (+0.004) | 41.58% (+0.01%) | 29.57% (-0.02%) | 22.29% (-0.02%) | 99.84% | 22.42% | 1222.3 | 1.168× |
| D (Stabilized)| $\lambda=8.0$ | 3.393 (+0.003) | 41.63% (+0.05%) | 29.59% (-0.01%) | 22.28% (-0.03%) | 99.83% | 22.41% | 1199.3 | 1.191× |

---

## 4. Regional Breakdown

Across all 7 evaluated regions, differences between the candidates are numerically negligible and not consistently directional:

### Regional Comparison (Lane L)

| Region | Metric | A (Legacy) | B (CurvGate $\tau=4$) | C (ConfGate $\tau=0.05$) | D (Stabilized $\lambda=4$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **All Pixels** | EPE / Bad0.5 / Bad2 | 3.477 / 41.75% / 22.41% | 3.479 / 41.75% / 22.43% | 3.478 / 41.74% / 22.42% | 3.485 / 41.79% / 22.45% |
| **Non-Occ (Noc)**| EPE / Bad0.5 / Bad2 | 2.038 / 35.56% / 15.76% | 2.038 / 35.57% / 15.75% | 2.037 / 35.56% / 15.77% | 2.044 / 35.61% / 15.80% |
| **Occluded (Occ)**| EPE / Bad0.5 / Bad2 | 14.406 / 88.75% / 72.92% | 14.425 / 88.73% / 73.11% | 14.425 / 88.64% / 72.92% | 14.436 / 88.76% / 72.91% |
| **Boundary (Bnd)**| EPE / Bad0.5 / Bad2 | 7.406 / 69.01% / 46.92% | 7.405 / 69.09% / 47.02% | 7.393 / 69.04% / 46.92% | 7.399 / 68.98% / 46.79% |
| **Non-Boundary** | EPE / Bad0.5 / Bad2 | 3.228 / 40.02% / 20.86% | 3.231 / 40.03% / 20.87% | 3.231 / 40.01% / 20.87% | 3.238 / 40.08% / 20.91% |
| **Low Texture** | EPE / Bad0.5 / Bad2 | 3.588 / 47.00% / 24.77% | 3.589 / 46.99% / 24.77% | 3.590 / 46.97% / 24.78% | 3.603 / 47.04% / 24.84% |
| **High Texture**| EPE / Bad0.5 / Bad2 | 3.331 / 34.84% / 19.31% | 3.334 / 34.88% / 19.35% | 3.331 / 34.87% / 19.31% | 3.330 / 34.90% / 19.30% |

### Regional Comparison (Lane H)

| Region | Metric | A (Legacy) | B (CurvGate $\tau=4$) | C (ConfGate $\tau=0.05$) | D (Stabilized $\lambda=4$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **All Pixels** | EPE / Bad0.5 / Bad2 | 3.390 / 41.57% / 22.31% | 3.390 / 41.61% / 22.30% | 3.398 / 41.61% / 22.30% | 3.395 / 41.58% / 22.29% |
| **Non-Occ (Noc)**| EPE / Bad0.5 / Bad2 | 2.113 / 35.36% / 15.89% | 2.113 / 35.42% / 15.88% | 2.115 / 35.40% / 15.89% | 2.114 / 35.38% / 15.87% |
| **Occluded (Occ)**| EPE / Bad0.5 / Bad2 | 12.967 / 88.18% / 70.42% | 12.971 / 88.09% / 70.44% | 13.018 / 88.16% / 70.43% | 12.999 / 88.09% / 70.45% |
| **Boundary (Bnd)**| EPE / Bad0.5 / Bad2 | 7.142 / 68.86% / 46.90% | 7.141 / 68.85% / 46.85% | 7.154 / 68.88% / 46.85% | 7.144 / 68.83% / 46.87% |
| **Non-Boundary** | EPE / Bad0.5 / Bad2 | 3.153 / 39.85% / 20.76% | 3.153 / 39.89% / 20.75% | 3.160 / 39.88% / 20.75% | 3.158 / 39.86% / 20.73% |
| **Low Texture** | EPE / Bad0.5 / Bad2 | 3.395 / 46.87% / 24.61% | 3.397 / 46.91% / 24.58% | 3.406 / 46.90% / 24.58% | 3.403 / 46.88% / 24.56% |
| **High Texture**| EPE / Bad0.5 / Bad2 | 3.383 / 34.61% / 19.30% | 3.382 / 34.65% / 19.30% | 3.386 / 34.65% / 19.32% | 3.384 / 34.63% / 19.30% |

---

## 5. Telemetry Diagnostics

Across all 15 Middlebury scenes (averaging ~345k to 420k candidate pixels per frame):
- **Applied Subpixel Ratio**: **98.14%** in Lane L, **98.27%** in Lane H.
- **Degenerate Denominator Count**: **0** occurrences across all scenes on this dataset.
- **Negative Curvature / Clamped Pixels**: **1.63%** in Lane L, **1.34%** in Lane H.
- **Mean Absolute Subpixel Shift $|\delta|$**: **0.1805 px** in Lane L, **0.1788 px** in Lane H.
- **Percentiles**: Median shift (p50) is **0.135 px**; 95th percentile (p95) is **0.445 px**.

---

## 6. Implementation Architecture

1. **Production Fast-Path Inlining**:
   `SubpixelEstimator::compute_legacy_delta(cm, c0, cp)` is a `static inline noexcept` function. Production WTA loops evaluate `is_fast_legacy` outside the pixel loop, executing direct formula inlining with zero per-pixel experimental dispatch, zero switch statements, and zero thread-local lookups.
2. **Internal Experimental Surface**:
   Candidate methods (CurvGate, ConfGate, Stabilized), thresholds, and telemetry collection remain strictly internal to the testing and benchmark harness via `ExperimentalSubpixelScope`, preventing experimental leakage into `PipelineConfig` or public headers.
3. **Synthetic Test Coverage**:
   `tests/test_subpixel.cpp` tests 7 distinct synthetic curve profiles (symmetric, left-biased, right-biased, flat, negative curvature, shallow basin, uniqueness thresholding), ensuring mathematically exact parity across all engines.
