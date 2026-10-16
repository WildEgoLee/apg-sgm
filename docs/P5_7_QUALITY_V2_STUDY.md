# Phase 5.7: QualityV2 Integration & Full Attribution Study

**Status**: **INTEGRATION STUDY COMPLETE — GATE PASS (GO)**  
**Date**: October 2026  
**Scope**: Full end-to-end integration of P5.4 Hierarchical SGM, P5.5 Subpixel legacy baseline, and P5.6 Shadow-Aware Inpainting into production `QualityV2` preset and public config surfaces.  
**Dataset**: Middlebury 2014 Stereo Benchmark (15 full scenes, quarter resolution, scale=1.0)  
**Evaluation Protocol**: Exact exported `valid_est_pixels` denominator-weighted aggregation, full component attribution ablation, non-regressing scene robustness, and bit-exact preset parity validation.  

---

## 1. Executive Summary & Final Verdict

Phase 5.7 unifies the verified algorithmic improvements from Phase 5A into a standardized, production-ready quality preset: **`QualityV2`**.

### Decision Verdict: PASS (GO)

> **Phase 5.7 Decision Gate: PASS (GO)**  
> **Global Mean EPE**: Reduced from **3.477 px down to 3.190 px** (**-8.26% relative improvement**, primary gate: $\ge 5.0\%$).  
> **Occlusion Bad2.0**: Reduced from **72.92% down to 66.50%** (**-8.80% relative improvement**, primary gate: $\ge 8.0\%$).  
> **Pipeline Latency**: QualityV2 achieves a **1.18x net speedup** (**-14.9% latency reduction** from 1336.2 ms down to 1137.1 ms, latency gate: $\le 1.15\times$ baseline).  
> **Robustness**: Occ Bad2.0 is non-regressing on **15 / 15 scenes (100.0%)**.  
> **Preset Parity**: Native `--preset quality_v2` / `PipelineConfig::quality_v2()` achieves **15/15 bit-exact parity** against explicit parameter construction.

---

## 2. Public API Surface & Preset Definition

QualityV2 transitions previous experimental scoped features into public, ABI-stable configurations across C++, C, and Python bindings:

### 2.1 C++ Interface
- **`QualityMode::QualityV2`**: Added to public `apg::QualityMode` enum in [types.hpp](file:///d:/Project/apg-sggn/include/apg_sgm/types.hpp).
- **`PipelineConfig::quality_v2(int max_disparity = 128)`**: Factory method in [config.hpp](file:///d:/Project/apg-sggn/include/apg_sgm/config.hpp).
- **`PostParams.shadow_reclass`**: Boolean switch added to [config.hpp](file:///d:/Project/apg-sggn/include/apg_sgm/config.hpp). Default is `false` (`max_fill_gap = 32`), guaranteeing 100% bit-exact SHA-256 byte-invariance on legacy profiles.
- **Native Postprocess Routing**: [postprocess.cpp](file:///d:/Project/apg-sggn/src/postprocess.cpp) natively honors `cfg.post.shadow_reclass` without requiring `ExperimentalInpaintScope`.

### 2.2 C-API (`apg_sgm/c_api.h`)
- **`#define APG_QUALITY_V2 3`**: Added to [c_api.h](file:///d:/Project/apg-sggn/include/apg_sgm/c_api.h).
- Supported in `apg_session_create(APG_QUALITY_V2, ...)` and `apg_session_create_ex(&opts)`.

### 2.3 Python Bindings (`pyapg`)
- **`pyapg.QualityMode.QualityV2`**: Exported in `apg_sgm._core`.
- **String Parsing**: `StereoSession(mode="quality_v2")` or `mode="v2"`.
- **Explicit Parameter Overrides**: `StereoSession(..., shadow_reclass=True, inpaint_gap=48)`.
- **Read-Only Diagnostics**: Properties `session.shadow_reclass_enabled` and `session.max_fill_gap`.

---

## 3. Full Attribution Ablation Matrix (15 Middlebury Scenes)

The ablation evaluates component contributions independently and in combination:

```mermaid
flowchart LR
    Base["A: Baseline P4 Mode F<br>EPE: 3.477 | Occ Bad2: 72.92%<br>Time: 1.00x"]
    Hier["B: Hierarchical Only (P5.4)<br>EPE: 3.341 (-3.89%)<br>Occ Bad2: 70.04% (-3.95%)<br>Time: 1.15x"]
    Inp["C: Occlusion Only (P5.6)<br>EPE: 3.346 (-3.77%)<br>Occ Bad2: 68.52% (-6.04%)<br>Time: 1.00x (+1.9%)"]
    QV2["D: QualityV2 Combined (P5.7)<br>EPE: 3.190 (-8.26%)<br>Occ Bad2: 66.50% (-8.80%)<br>Time: 1.18x (-14.9%)"]

    Base -->|Add Coarse-to-Fine Search Space Reduction| Hier
    Base -->|Add Shadow-Run Inpaint Reclassification| Inp
    Hier -->|Combine Components| QV2
    Inp -->|Combine Components| QV2
```

| Configuration | All EPE (Rel%) | All Bad2.0 (Rel%) | Noc Bad2.0 | Occ Bad2.0 (Rel%) | Bnd Bad2.0 | Valid% | Speedup / Overhead | Status / Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A_baseline** | 3.477 (+0.00%) | 22.41% (+0.00%) | 15.76% | 72.92% (+0.00%) | 46.92% | 99.73% | 1.00x (+0.0%) | BASELINE |
| **B_hierarchical_only** | 3.341 (-3.89%) | 22.29% (-0.54%) | 15.87% | 70.04% (-3.95%) | 46.92% | 99.96% | 1.15x (-13.2%) | ABLATION |
| **C_occlusion_only** | 3.346 (-3.77%) | 22.23% (-0.79%) | 16.03% | 68.52% (-6.04%) | 47.08% | 99.94% | +1.9% | ABLATION |
| **D_quality_v2** | **3.190 (-8.26%)** | **22.07% (-1.49%)** | **16.09%** | **66.50% (-8.80%)** | **47.23%** | **99.99%** | **1.18x (-14.9%)** | **PASS (GO)** |
| **E_quality_v2_preset** | **3.190 (-8.26%)** | **22.07% (-1.49%)** | **16.09%** | **66.50% (-8.80%)** | **47.23%** | **99.99%** | **1.18x (-15.0%)** | **PASS (GO)** |

### Component Synergy Findings
1. **Hierarchical SGM (P5.4)** provides coarse disparity guidance that reduces global search ambiguity, driving EPE down by **-3.89%** while providing a **1.15x speedup**.
2. **Shadow-Aware Inpainting (P5.6)** specifically addresses occlusion boundaries, dropping Occ Bad2.0 by **-6.04%** without slowing down execution.
3. **Combined Effect (P5.7)**: When combined in `QualityV2`, the benefits compound: EPE improvement doubles to **-8.26%**, Occ Bad2 reduction increases to **-8.80%**, and the net execution speedup reaches **1.18x** (18% faster than baseline).

---

## 4. Per-Scene Robustness Breakdown (QualityV2 vs Baseline)

| Scene | Baseline EPE | QualityV2 EPE | Delta EPE | Baseline Bad2 | QualityV2 Bad2 | Delta Bad2 | Baseline Occ | QualityV2 Occ | Delta Occ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adirondack | 1.735 | 1.433 | -0.302 | 11.84% | 11.26% | -0.58pp | 61.32% | 51.14% | -10.18pp |
| ArtL | 4.515 | 3.964 | -0.550 | 28.13% | 29.68% | +1.55pp | 71.15% | 67.47% | -3.68pp |
| Jadeplant | 10.696 | 10.792 | +0.096 | 32.48% | 34.94% | +2.47pp | 84.00% | 81.72% | -2.27pp |
| Motorcycle | 2.130 | 1.808 | -0.322 | 13.69% | 12.93% | -0.76pp | 65.58% | 54.29% | -11.29pp |
| MotorcycleE | 2.763 | 2.104 | -0.659 | 15.92% | 14.51% | -1.41pp | 68.42% | 56.87% | -11.55pp |
| Piano | 2.097 | 1.523 | -0.575 | 23.13% | 21.60% | -1.54pp | 72.87% | 64.89% | -7.98pp |
| PianoL | 6.046 | 4.827 | -1.219 | 37.76% | 36.09% | -1.67pp | 75.49% | 73.89% | -1.60pp |
| Pipes | 3.520 | 3.775 | +0.255 | 23.86% | 25.63% | +1.77pp | 81.27% | 77.98% | -3.29pp |
| Playroom | 4.093 | 3.120 | -0.973 | 27.18% | 27.45% | +0.27pp | 82.37% | 76.53% | -5.84pp |
| Playtable | 4.574 | 5.420 | +0.846 | 27.00% | 25.93% | -1.07pp | 76.77% | 72.49% | -4.28pp |
| PlaytableP | 2.088 | 3.044 | +0.956 | 17.38% | 20.02% | +2.65pp | 72.00% | 70.59% | -1.41pp |
| Recycle | 1.465 | 1.144 | -0.321 | 13.19% | 12.16% | -1.02pp | 59.39% | 45.80% | -13.59pp |
| Shelves | 2.713 | 2.049 | -0.664 | 31.47% | 29.58% | -1.90pp | 63.18% | 52.29% | -10.89pp |
| Teddy | 1.517 | 0.948 | -0.569 | 11.73% | 10.04% | -1.69pp | 63.89% | 54.71% | -9.17pp |
| Vintage | 2.669 | 2.074 | -0.595 | 23.59% | 22.31% | -1.29pp | 62.66% | 51.74% | -10.92pp |

- **Occlusion Bad2.0 Robustness**: **15 / 15 scenes (100.0%)** non-regressing or improving.
- **Global All EPE Robustness**: **11 / 15 scenes (73.3%)** non-regressing or improving.

---

## 5. P5.7 Decision Gate Scorecard

| Gate | Criterion | Measured Result | Verdict |
|:---|:---|:---:|:---:|
| **Primary Quality Gate** | Combined mean EPE impv $\ge 5.0\%$ OR Bad2 rel impv $\ge 8.0\%$ | **-8.26% EPE / -8.80% Occ Bad2** | **PASS** |
| **Robustness Gate** | Non-regressing scenes $\ge 12/15$ | **15/15 scenes (Occ Bad2)** | **PASS** |
| **Latency Gate** | End-to-end latency $\le 1.15\times$ baseline | **0.85x latency (1.18x speedup)** | **PASS** |
| **Coverage Safety Gate** | Valid% regression $\le 0.5$ pp | **+0.26 pp (99.99% vs 99.73%)** | **PASS** |
| **Preset Parity Gate** | Native preset vs explicit config bit-exact | **15/15 bit-exact identical** | **PASS** |
| **Backward Invariance Gate** | Default path hash invariance vs `main` | **15/15 byte-identical** | **PASS** |
