# Phase 5.3 Feasibility Study Report: Hierarchical SGM Search Space Reduction

## 1. Executive Summary & Verdict

Phase 5.3 evaluates the algorithmic and systemic feasibility of **2-Level Hierarchical SGM (Coarse-to-Fine Search Space Reduction)** for APG-SGM on the canonical 15-scene Middlebury benchmark using the F-mode packed backend.

```text
Base Commit      : 7d6844b (feat(eval): Polish evaluation integrity, regional aggregation, and tri-gate protocol (P5.0a))
Evaluation Scope : Middlebury 2014 full-resolution 15 scenes (F-mode packed backend, 16 threads)
Gate Verdict     : DUAL GO (Performance GO PASS, Memory GO PASS, Quality Tri-Gate PASS)
```

### Decision Gate Verification

| Gate | Criterion | P5.0a Baseline | Candidate ($R=4$) | Delta / Ratio | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Performance GO (Speedup)** | Speedup $\frac{T_B}{T_C} - 1 \ge 15\%$ | 1365.38 ms | 1152.99 ms | **+18.42% Aggregate (1.1842×)**<br>**+18.28% Geomean (1.1828×)** | **PASS** |
| **Performance GO (Latency)** | Latency Reduction $1 - \frac{T_C}{T_B}$ | 1365.38 ms | 1152.99 ms | **15.56% Aggregate**<br>**15.46% Geomean** | **PASS** |
| **Memory GO (Corrected)** | Peak Workspace $\ge 30\%$ drop | 143.6 MB | 36.0 MB | **-74.93% (-107.6 MB peak)** | **PASS** |
| **Quality: EPE** | EPE regression $\le 1.0\%$ | 3.477 px | 3.390 px | **-2.50% (better)** | **PASS** |
| **Quality: Bad2 (Valid)** | Bad2 regression $\le 0.5$ pp | 22.41% | 22.31% | **-0.10 pp (better)** | **PASS** |
| **Quality: Valid%** | Valid% drop $\le 0.5$ pp | 99.73% | 99.83% | **+0.10 pp (higher)** | **PASS** |
| **Quality: Bad2_all** | Bad2_all degradation $\le 0$ pp | 22.62% | 22.44% | **-0.18 pp (better)** | **PASS** |

---

## 2. Architecture & Encapsulation Integrity

### 2.1 Public API & ABI Preservation
- **Public Headers**: `include/apg_sgm/buffers.hpp` and `include/apg_sgm/config.hpp` remain **100% bit-exact identical** to `7d6844b` (P5.0a baseline).
- **Zero Struct/Layout Mutation**: `sizeof(PipelineBuffers)` is unchanged; no public fields (`coarse_buffers`) are exposed; value-like copy semantics are preserved.
- **Thread-Local Experimental Scope**: Feasibility parameters ($S=2$, radius $R \in \{4, 8, 12, 16\}$, fallback policy) are passed via thread-local RAII context (`ExperimentalHierarchicalScope`) and benchmark CLI arguments (`--hierarchical`, `--hierarchical-radius`, `--hierarchical-fallback`).

### 2.2 Exhaustive Memory Instrumentation
Coarse memory is encapsulated within a local `PipelineBuffers cbuf` inside `HierarchicalStereo::estimate_search_range`. Memory high-water sampling is passed via an explicit extra-bytes callback:
```cpp
auto sample_workspace_extra = [&](size_t coarse_bytes) {
    workspace_high_water = std::max(workspace_high_water, out.allocated_bytes() + coarse_bytes);
};
```
The sampler is invoked comprehensively at all lifecycle transitions:
1. `cbuf.left / right` 2x downsampled allocation;
2. `cost_c.compute_aux(...)` coarse gradient/census allocation;
3. `cbuf.range.allocate(...)` coarse search range;
4. `cbuf.packed_cost.allocate_for_overwrite(...)` coarse packed cost volume;
5. `agg_c.aggregate_packed(...)` coarse cross-aggregation;
6. **`sgm_c.optimize_packed(...)` coarse SGM accumulator allocation (coarse cost volume and accumulator simultaneously alive)**;
7. `sgm_c.winner_take_all_packed(...)` coarse WTA;
8. `conf_c.estimate(...)` coarse confidence;
9. `post_c.left_right_check(...)` and hole filling;
10. Immediately before and after releasing coarse temporary volumes;
11. At the completion of guided range construction before returning to fine SGM matching.

---

## 3. Multi-Radius Quality Exploration (Middlebury 15 Scenes)

Evaluated under the canonical P5.0a regional evaluation protocol (`tools/summarize_p5_baseline.py`) with exact valid pixel weighting:

| Configuration | Peak MB | Avg States/px | Valid % | EPE (px) | Bad2 (Valid) % | Bad2_all % | Noc Bad2 % | Occ Bad2 % | Bnd Bad2 % | LowTex Bad2 % | HighTex Bad2 % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P5.0a Baseline** | 143.6 | 55.80 | 99.73% | 3.477 | 22.41% | 22.62% | 15.76% | 72.92% | 46.92% | 24.77% | 19.31% |
| **Candidate $R=4$** | **36.0** | **10.35** | **99.83%** | **3.390** | **22.31%** | **22.44%** | 15.89% | 70.42% | 46.90% | 24.61% | 19.30% |
| **Candidate $R=8$** | 45.7 | 17.76 | 99.83% | 3.422 | 22.62% | 22.76% | 16.13% | 71.34% | 46.83% | 24.91% | 19.62% |
| **Candidate $R=12$**| 56.3 | 25.18 | 99.82% | 3.462 | 22.67% | 22.81% | 16.15% | 71.64% | 46.68% | 24.94% | 19.69% |
| **Candidate $R=16$**| 66.5 | 32.40 | 99.82% | 3.512 | 22.65% | 22.79% | 16.09% | 71.89% | 46.55% | 24.90% | 19.70% |

### Rationale for $R=4$:
$R=4$ is the pareto-optimal configuration:
1. **Best Accuracy**: Delivers the lowest EPE (3.390 px) and lowest Bad2 (22.31%), outperforming baseline and wider radii.
2. **Noise Suppressing**: Restricting active fine search hypotheses to the neighborhood of the coarse disparity envelope eliminates spurious false minima in low-texture and occluded zones, reducing Occlusion Bad2 from 72.92% to 70.42%.
3. **Lowest Footprint**: Average states per pixel drop from 55.80 to 10.35 (an 81.5% reduction in dynamic programming transitions). Peak workspace drops from 143.6 MB to 36.0 MB (-74.93%).

---

## 4. Matched Interleaved Performance Benchmark ($B1 \to C1 \to C2 \to B2$)

Conducted with 16 threads, warmup 2, repeat 5, packed F-mode backend across all 15 Middlebury scenes on identical hardware:

| Scene | B1 (ms) | B2 (ms) | B_mean (ms) | C1 (ms) | C2 (ms) | C_mean (ms) | Delta (ms) | Speedup ($\frac{T_B}{T_C}$) | Latency Reduction ($1 - \frac{T_C}{T_B}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adirondack** | 102.31 | 103.47 | 102.89 | 87.76 | 91.76 | 89.76 | -13.13 | 1.146× | 12.76% |
| **ArtL** | 27.29 | 29.56 | 28.42 | 24.70 | 24.27 | 24.48 | -3.94 | 1.161× | 13.86% |
| **Jadeplant** | 115.50 | 117.82 | 116.66 | 88.58 | 87.80 | 88.19 | -28.47 | **1.323×** | **24.40%** |
| **Motorcycle** | 91.70 | 95.74 | 93.72 | 82.67 | 81.56 | 82.12 | -11.60 | 1.141× | 12.38% |
| **MotorcycleE** | 101.96 | 104.68 | 103.32 | 90.55 | 93.55 | 92.05 | -11.27 | 1.122× | 10.91% |
| **Piano** | 83.97 | 91.35 | 87.66 | 76.62 | 81.32 | 78.97 | -8.69 | 1.110× | 9.91% |
| **PianoL** | 105.05 | 108.26 | 106.66 | 94.45 | 92.27 | 93.36 | -13.30 | 1.142× | 12.47% |
| **Pipes** | 89.05 | 93.71 | 91.38 | 76.43 | 79.05 | 77.74 | -13.64 | 1.175× | 14.93% |
| **Playroom** | 94.66 | 98.42 | 96.54 | 85.46 | 82.56 | 84.01 | -12.53 | 1.149× | 12.98% |
| **Playtable** | 92.91 | 93.87 | 93.39 | 75.45 | 75.27 | 75.36 | -18.03 | 1.239× | 19.31% |
| **PlaytableP** | 90.45 | 91.94 | 91.19 | 74.00 | 73.69 | 73.84 | -17.35 | 1.235× | 19.03% |
| **Recycle** | 97.50 | 96.11 | 96.81 | 81.03 | 81.23 | 81.13 | -15.68 | 1.193× | 16.19% |
| **Shelves** | 99.88 | 98.91 | 99.39 | 85.84 | 86.85 | 86.34 | -13.05 | 1.151× | 13.13% |
| **Teddy** | 35.35 | 40.25 | 37.80 | 31.06 | 31.64 | 31.35 | -6.45 | 1.206× | 17.06% |
| **Vintage** | 119.57 | 119.51 | 119.54 | 90.78 | 97.78 | 94.28 | -25.26 | 1.268× | 21.13% |
| **Total / Mean** | **1347.15** | **1383.60** | **1365.38** | **1145.38** | **1160.60** | **1152.99** | **-212.38** | **1.184×** | **15.56%** |

### Benchmark Metrics Summary:
- **Aggregate Speedup**: **1.1842× (+18.42%)**
- **Aggregate Latency Reduction**: **15.56%**
- **Geomean Speedup**: **1.1828× (+18.28%)**
- **Geomean Latency Reduction**: **15.46%**
- **Consistency**: 15 / 15 scenes exhibit $>1.0\times$ speedup (range 1.110× to 1.323×).

---

## 5. Search Space Accounting Telemetry ($R=4$)

| Metric | Measured Value | Analysis |
| :--- | :---: | :--- |
| **`avg_states_per_pixel`** | **10.35** | Down from 55.80 on baseline (-81.5% search hypotheses). |
| **`coarse_valid_ratio`** | **99.42%** | High coverage from coarse matching with hole-filling. |
| **`fallback_pixel_ratio`** | **0.44%** | Only 0.44% of pixels lack valid 2x2 coarse neighbors. |
| **`full_range_pixel_ratio`** | **0.37%** | Only 0.37% of pixels fall back to full search range $[D_{\min}, D_{\max}]$. |

---

## 6. Algorithmic Scope & Fallback Policy Characterization

### Validated Fallback Mechanism
The feasibility experiment specifically validated:
> **$2\times$ coarse-disparity guided search space reduction with invalid-neighbor spatial extrapolation (`NearestValid`) is highly effective and robust.**

Fallback is triggered when all 4 coarse neighbors in the 2x2 upsampling footprint are invalid ($d < 0$). In this event, a spatial window ($K=4$ coarse pixels) searches for the nearest valid estimate to anchor the search range.

### Clarification on Confidence-Aware Hierarchy
The current feasibility prototype evaluates coarse confidence via `ConfidenceEstimator` but does **not** yet modulate the search radius $R$ dynamically based on confidence thresholds. 

---

## 7. Known Limitations & Roadmap for Phase 5.4

1. **Static Search Radius**: The search radius $R=4$ is currently spatially uniform across all matchable pixels. Phase 5.4 should explore confidence-aware radius expansion:
   - High confidence $\to R=4$;
   - Low confidence / boundary $\to$ widened radius $R=8$ or full-range fallback.
2. **SIMD Kernel Specialization**: Fine SGM matching currently operates on arbitrary dynamic interval ranges $[d_{\min}(x, y), d_{\max}(x, y)]$. Because $R=4$ yields a narrow bounded hypothesis width ($2R+1 = 9$ disparities), Phase 5.5 can implement dedicated AVX2 path kernels specialized for fixed-width hypothesis blocks to accelerate SGM dynamic programming further.
3. **Public API Graduation (Phase 5.4)**: Formalize `HierarchicalConfig` within `PipelineConfig` and expose Python properties on `StereoSession`.
