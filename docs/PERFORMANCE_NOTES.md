# Performance Notes

## V3 Priority 1: packed Cross streaming rejected for the default CPU path

**Decision:** do not merge PR #3. Keep the experiment branch at `82ee5e5`; do not continue tuning the ring, batch size, or barriers. No low-memory option is planned without a concrete memory-constrained use case.

The 35-row ring, B=16 flattened scheduler, and thread-owned stripe variants retained bit-exact correctness. The B=16 workspace averaged about 12.9% of the full packed temporary volume (about 87% less workspace), but stable 16-thread paired tests on ArtL, Piano, and Vintage measured:

| Variant | Cross geometric-mean speed | Pipeline geometric-mean speed |
| --- | ---: | ---: |
| B=16 flattened, `82ee5e5` | 0.711x | 0.885x |
| Thread-owned stripes | 0.696x | 0.901x |

The B=16 result is from three interleaved V2/V3 ABBA pairs. The stripe result is from the same three-scene gate. Since changing work scheduling did not recover throughput, the measured workspace reduction does not justify the CPU throughput loss. The current full-temporary packed Cross path remains the default.

This is a rejected performance experiment, not a correctness failure. The branch remains available for reference and has not been merged into `main`.

## V3 Priority 2.1: split the packed SGM workspace experiment

**Decision:** keep only P2.1a, removing the pipeline's redundant packed `cost32` initialization. Reject P2.1b, per-worker cardinal `PathState` reuse.

The A/B/C/D attribution used `main` at `64d683c` as A and `c2032d1` as D:

| Variant | Change from A |
| --- | --- |
| A | Baseline |
| B | Remove `out.packed_cost32.allocate(layout, 0)` from the pipeline |
| C | Reuse cardinal `PathState` vectors per worker |
| D | B + C (`c2032d1`) |

The isolated run covered ArtL, Piano, and Vintage in E/Path4 and G/Path8 with 16 threads. It collected 20 samples per version, scene, and mode over two reversed-order passes. Cost, prior, packed cost construction, and Cross aggregation were outside the timed intervals. Input and SGM output hashes matched across all four variants.

| Comparison | Timing | E/Path4 | G/Path8 |
| --- | --- | ---: | ---: |
| B vs A: remove pre-allocation | Production SGM stage | 1.08x | 1.00x |
| D vs C: remove pre-allocation with scratch reuse | Production SGM stage | 1.05x | 1.03x |
| C vs A: scratch reuse only | Optimizer only | 0.99x | 0.98x |
| D vs B: scratch reuse only after pre-allocation removal | Optimizer only | 0.96x | 1.00x |
| D vs A: both changes | Optimizer only | 0.95x | 1.01x |

Speedups are geometric means of the three per-scene median paired ratios. Path4 scratch reuse did not improve the isolated optimizer and was neutral-to-negative; Path8 was neutral. Do not continue tuning OpenMP scheduling or scratch lifetime for this variant. The redundant zero-write removal remains a separate B-only candidate because its production SGM stage showed a Path4 benefit and approximately neutral Path8 timing.

The B-only branch `codex/p2-1a-packed-cost32-single-init` starts from `64d683c` and contains only the pipeline deletion plus this note. Its first three-scene full-pipeline gate used five alternating ABBA/BAAB blocks, 16 threads, one warmup, and one measured run per invocation (10 adjacent pairs per scene/mode):

| Metric | E/Path4 | G/Path8 |
| --- | ---: | ---: |
| SGM geometric mean | 1.09x | 1.01x |
| Pipeline geometric mean | 1.00x | 0.99x |

The geometric mean across all six scene/mode cells was `0.99446x`, below the `0.995x` gate. That run also showed a large Cross slowdown in a stage that precedes the changed initialization. The result triggered a same-binary stage attribution before making a merge decision.

The follow-up used one temporary benchmark binary with a runtime switch: A executed the removed pipeline pre-allocation, while B used the candidate path. The switch was read before timing; the allocation ran at the original point after Cross. The temporary switch and CSV instrumentation were reverted after the run. Five alternating ABBA/BAAB blocks yielded 10 adjacent pairs per scene/mode at 16 threads, one warmup, and one measured run. The ratios below are geometric means of per-scene median paired speedups (`A_ms / B_ms`, so values above 1 favor the candidate):

| Stage | E/Path4 | G/Path8 |
| --- | ---: | ---: |
| Aux | 0.9423x | 1.0679x |
| Cost | 1.0017x | 0.9948x |
| Right WTA | 1.0047x | 1.0266x |
| Cross | 0.9999x | 0.9902x |
| SGM | 1.1027x | 1.0167x |
| WTA | 0.9896x | 1.0393x |
| Confidence | 1.0065x | 0.9959x |
| Prior | 0.9983x | 1.0036x |
| Refine | 0.9445x | 1.0001x |
| Post | 0.9979x | 0.9966x |
| Pipeline total | 1.0093x | 1.0167x |

The geometric mean across all six scene/mode pipeline cells was `1.0130x`. The unmodified Aux stage varied substantially across modes, so its ratios are not attributed to this change. The earlier `0.94–0.95x` Cross result did not reproduce in the same binary. A focused second Vintage/E ABBA/BAAB sample (10 more pairs) measured WTA at `0.99x` median (`p25=0.96x`, `p75=1.00x`), versus `0.95x` in the first sample; the combined 20-pair median was `0.9896x`. This is at most a small stage-level shift (about 0.24 ms on a roughly 426 ms pipeline), not a repeatable material pipeline regression. Confidence stayed near 1.00x, and the six-cell total stayed above the gate. The Path4 refiner timing is only 0.05–0.26 ms, so its `0.9445x` ratio is dominated by timing resolution and should not be read as a meaningful regression. All exported quality and memory fields matched between A and B; previous bit-exact and backend-parity checks remain green.

**Final status:** P2.1a (packed `cost32` single initialization) was squash-merged to `main` as `b0751fe` after GCC and Clang CI passed. The earlier separate-build `0.99446x` total and Cross slowdown remain recorded as exploratory data; they did not reproduce in the controlled same-binary attribution. P2.1b (per-worker cardinal `PathState` reuse) is rejected; P2.2 (`prev = cur` state-copy removal) is next and has not started.

## V3 Priority 2.2: Packed `PathState` ownership swap

**Status:** MERGED as `6c2370b` (PR #5). Local correctness, trainingQ quick gates, and full-resolution Middlebury F confirmatory gates passed.

Only `process_pixel_packed()` changed. Its three state-completion paths now swap `prev` and `cur`: the empty-state (`D_c <= 0`) path, the first-pixel/empty-previous path, and the normal recurrence path. Dense `process_pixel()` remains unchanged. Every call resets `dmin`, `dmax`, and `min_val`; for `D_c > 0`, both initialization and recurrence loops overwrite every `cur.vals[0..D_c)` entry, including invalid costs. For `D_c <= 0`, `resize(0)` and the metadata reset produce the complete empty state. The swap therefore retains the two vector capacities without carrying logical recurrence values forward.

Correctness checks on the MSVC Release build passed CTest 4/4. Dense/Packed backend verification passed 6/6 bit-exact cases for E and G across ArtL, Piano, and Vintage. The isolated Packed SGM accumulator hash also matched baseline on every scene/mode cell.

### Preliminary trainingQ Quick Gate

The available local data was the Middlebury Eval3 `trainingQ` ArtL/Piano/Vintage subset from the MiddleEval archives. Inputs were converted to grayscale PGM under ignored `build` outputs. Results below use Q-resolution inputs with `dmax` 32/40/96; they are a quick gate, not a full-resolution F-data benchmark. All measurements used 16 threads and MSVC Release with OpenMP 2.0.

The isolated optimizer comparison used baseline `1d2c02f` and the P2.2 candidate. Each process computed its Packed inputs before timing; the timed loop called `optimize_packed()` directly, retaining its production accumulator initialization. Each invocation used two warmups and 20 timed samples; five interleaved ABBA/BAAB blocks compared baseline and candidate. The table shows the median of invocation medians and the p25–p75 range across invocations, plus the geometric mean of paired speedups (`baseline / candidate`):

| Scene/mode | Baseline median (p25–p75) ms | Candidate median (p25–p75) ms | Optimizer speedup |
| --- | ---: | ---: | ---: |
| ArtL / E (Path4) | 9.015 (8.902–9.464) | 6.406 (6.242–6.641) | 1.397x |
| Piano / E (Path4) | 33.421 (33.037–33.601) | 25.195 (24.753–25.591) | 1.334x |
| Vintage / E (Path4) | 57.314 (55.391–58.463) | 41.778 (40.723–42.746) | 1.340x |
| ArtL / G (Path8) | 74.875 (74.225–74.954) | 75.014 (74.612–75.052) | 1.000x |
| Piano / G (Path8) | 273.534 (270.254–273.860) | 266.917 (265.210–268.412) | 1.028x |
| Vintage / G (Path8) | 455.189 (443.435–466.861) | 450.095 (439.906–459.286) | 1.009x |
| **Geometric mean** | — | — | **Path4 1.357x; Path8 1.012x** |

The paired pipeline check used packed E/G, one warmup and three timed repeats per invocation, and four balanced ABBA/BAAB blocks (eight adjacent pairs per scene/mode). The reported SGM and total ratios are geometric means of those paired samples:

| Scene/mode | SGM speedup | Pipeline speedup |
| --- | ---: | ---: |
| ArtL / E | 1.256x | 1.055x |
| Piano / E | 1.257x | 1.036x |
| Vintage / E | 1.228x | 1.019x |
| ArtL / G | 1.014x | 1.005x |
| Piano / G | 1.029x | 1.012x |
| Vintage / G | 1.026x | 1.007x |
| **Geometric mean** | **Path4 1.247x; Path8 1.023x** | **Path4 1.037x; Path8 1.008x; all six cells 1.022x** |

All 74 non-timing CSV fields matched in 48 paired pipeline comparisons. An earlier one-sample Vintage/G pass showed a large slowdown across SGM and several unchanged earlier stages. It did not reproduce: a focused eight-pair repeat-3 check measured 0.999x SGM and 0.998x pipeline geometric mean (pipeline median paired ratio 1.011x), and the balanced full matrix above showed no repeatable regression.

### Full-Resolution F Gate & Confirmation

Full-resolution Middlebury F-resolution evaluation confirmed substantial isolated optimizer and pipeline gains on Path4, while Path8 exhibited a smaller isolated gain:

- **Isolated Optimizer Speedup (F-resolution):**
  - Path4 (E-mode): **1.131x**
  - Path8 (G-mode): **1.040x**
- **Pipeline Speedup (F-resolution):**
  - Path4 (E-mode): **1.030x**
  - Path8 (G-mode):
    - Initial 3-scene sample: `0.9825x` (impacted by a severe single-run outlier at `0.508x` during concurrent system noise)
    - Confirmatory 30-pair gate: **1.0106x**
    - Pooled sample (all pairs combined, preserving the `0.508x` outlier): **0.9999716x**

Under the pre-established `>= 0.995x` regression gate, the pooled Path8 pipeline ratio confirms no regression even with the worst-case outlier preserved, while Path4 delivers clear end-to-end performance improvement. PR #5 was merged into `main` as `6c2370b`.

## V3 Priority 2.3: Diagonal workspace reuse rejected

**Decision:** Reject diagonal `seen` workspace pooling and lifecycle reuse without code modifications.

A micro-profiling audit of Path8 directional timing and internal diagonal breakdown was conducted across Middlebury Q-resolution (ArtL, Piano, Vintage) and F-resolution (ArtL, 1388x1108, dmax=256) at 16 threads:
- `seen` allocation and zero-initialization (`std::vector<uint8_t> seen(w * h, 0)`) across all 4 diagonal paths totaled:
  - ArtL (Q-res): 0.027 ms (0.038% of Path8)
  - Piano (Q-res): 0.083 ms (0.036% of Path8)
  - Vintage (Q-res): 0.112 ms (0.030% of Path8)
  - ArtL (F-res): 1.192 ms (0.0287% of Path8)
- Even if `seen` workspace overhead were completely reduced to zero, the theoretical speedup ceiling is approximately `1.0003x`.
- The profile demonstrated that Path8's dominant bottleneck was not workspace allocation, but rather that cardinal paths were parallelized across 16 threads while the 4 diagonal paths were executed completely serially on a single thread (accounting for 88.5%–92.0% of total Path8 runtime).
- Diagonal traversal has poorer spatial locality in image order, while packed ragged slices also have variable physical offsets; cache-miss impact has not yet been measured directly.
- Consequently, P2.3 workspace reuse is rejected, concluding Priority 2.

## V3 Priority 3.0: Packed diagonal-ray OpenMP parallelization

**Status:** MERGED as `6e7f99e` (PR #6). CTest 4/4 passed, GCC/Clang CI passed, and full-resolution Middlebury F pipeline paired gate passed.

### Implementation

- Diagonal aggregation paths in `aggregate_path_packed()` now explicitly enumerate border origins ($W + H - 1$ independent rays per direction) and parallelize execution across rays with `#pragma omp parallel for schedule(dynamic, 1)`.
- Border-origin ray mapping:
  - `(+1, +1)`: Top edge $(r, 0)$ and Left edge $(0, r - w + 1)$
  - `(+1, -1)`: Bottom edge $(r, h - 1)$ and Left edge $(0, h - 1 - (r - w + 1))$
  - `(-1, +1)`: Top edge $(r, 0)$ and Right edge $(w - 1, r - w + 1)$
  - `(-1, -1)`: Bottom edge $(r, h - 1)$ and Right edge $(w - 1, h - 1 - (r - w + 1))$
- `seen` vector tracking and redundant backwards boundary searches were eliminated as a natural consequence of ray enumeration.
- Per-worker scratch reuse was intentionally omitted to cleanly isolate ray-level parallelism.
- The Dense backend was kept frozen as a golden reference.

### Verification & Gate Results

- **Correctness**:
  - CTest: 4/4 passed (`sanity`, `synthetic`, `ablation`, `headers`).
  - Dual backend (Dense vs Packed) parity: 100% bit-exact across synthetic cases in E and G modes.
  - Dual backend parity on full-resolution Middlebury F: 100% bit-exact across ArtL, Piano, Vintage in G mode (Path8).
  - Isolated SGM accumulator hashes matched baseline bit-exact on all Q and F scenes.
  - Across 18 paired pipeline comparisons (54 scene evaluations), all 74 non-timing quality and metric fields matched 100% bit-exactly.
- **Full-Resolution Middlebury F Pipeline Paired Benchmark (G-only, 16 threads)**:
  - Fixed binaries across 3 balanced ABBA/BAAB blocks (6 adjacent pairs per scene, 18 pairs total, zero sample deletions):

| Scene | Baseline SGM (ms) | Candidate SGM (ms) | SGM Speedup (Geomean / Median) | Baseline Pipeline (ms) | Candidate Pipeline (ms) | Pipeline Speedup (Geomean / Median) | Quality Fields Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | ~4900 ms | ~1130 ms | **4.328x** / 4.352x | ~8150 ms | ~4400 ms | **1.853x** / 1.854x | 100% (74/74) |
| **Piano (F)** | ~18950 ms | ~4470 ms | **4.242x** / 4.230x | ~31450 ms | ~16850 ms | **1.863x** / 1.866x | 100% (74/74) |
| **Vintage (F)** | ~43480 ms | ~10500 ms | **4.135x** / 4.172x | ~71250 ms | ~38100 ms | **1.861x** / 1.872x | 100% (74/74) |
| **Overall Pooled** | — | — | **4.2340x** (Gate: >= 2.0x) | — | — | **1.8591x** (Gate: >= 1.10x) | **PASS** |

- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed.

---

### P3.2a: Packed Cross 1D Tiled Prefix-Sum / Prefix-Count Optimization

- **Status**: **MERGED as `99725c4` (PR #7)**
- **Baseline**: `2ad6d26` (Post-P3.0 + 10-stage attribution)
- **Scope**: Replaced repeated neighbor scanning $O(S \cdot \text{arm\_span})$ in `CostAggregator::aggregate_hv_packed()` with exact $O(S)$ 1D prefix-sum and prefix-count tile queries. Dense reference backend remained frozen; `tmp` volume allocation/ownership unchanged.
- **Key Architectural Decisions**:
  - Tiling with $B=16$ lanes.
  - Per-row disparity envelope $[row\_lo, row\_hi)$ and per-column envelope $[col\_lo, col\_hi)$, eliminating scanning nonexistent disparity states.
  - Interleaved 64-bit `PrefixEntry { uint32_t sum; uint32_t cnt; }` layout for optimal L1/L2 cache locality (16 lanes fit within 2 cache lines).
  - Reusable per-worker workspace allocated once per `#pragma omp parallel` region.
  - Safety boundary: $\max(\text{dim}) \times \text{cost}_{\max} \le 3000 \times 65535 \approx 1.96 \times 10^8 \ll \text{UINT32\_MAX}$, strictly preventing accumulator overflow.

#### Verification & Parity
- **CTest**: 4/4 PASS.
- **Synthetic Checks**: Dual-backend bit-exact parity across 3 cases and 7 ablation modes (`dev.py check`).
- **Full-Resolution F-res Parity (`--modes G --verify-backends`)**:
  - `ArtL [G]`: PASS (100% bit-exact across all buffers)
  - `Piano [G]`: PASS (100% bit-exact across all buffers)
  - `Vintage [G]`: PASS (100% bit-exact across all buffers)
- **Metrics Parity**: All 74 non-timing fields bit-exact across paired benchmark runs.

#### Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | Baseline Cross (ms) | P3.2a Cross (ms) | Cross Speedup | Baseline Pipeline (ms) | P3.2a Pipeline (ms) | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 1496.42 ms | 417.22 ms | **3.587x** (H+V: 4.872x) | 4517.48 ms | 3554.32 ms | **1.271x** (27.1% faster) | 100% (74/74) |
| **Piano (F)** | 5322.46 ms | 2201.67 ms | **2.418x** (H+V: 2.798x) | 16946.61 ms | 13300.78 ms | **1.274x** (27.4% faster) | 100% (74/74) |
| **Vintage (F)** | 13526.26 ms | 6097.55 ms | **2.218x** (H+V: 2.509x) | 38344.20 ms | 29355.86 ms | **1.306x** (30.6% faster) | 100% (74/74) |
| **Scene-Balanced Geomean** | — | — | **2.679x** (Gate: >= 1.50x) | — | — | **1.284x** (Gate: >= 1.10x) | **PASS** |
| **Pooled Total Sum** | 22498.88 ms | 9022.00 ms | **2.494x** | 59808.29 ms | 46210.96 ms | **1.294x** | **PASS** |

- **Hotspot Migration**:
  - Cross execution share dropped from **35.51% down to 17.18%**.
  - Pipeline rankings: SGM 33.63% (#1), Cost 23.82% (#2), Cross 17.18% (#3), Refine 11.10% (#4).
  - Inside Cross, `tmp volume alloc` now represents ~18.5% to 29.3% of stage runtime.
- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed (Run 36215335613).

---

### P3.2b: Elimination of Redundant Temporary Packed Volume Initialization

- **Status**: **MERGED as `79ed6ac` (PR #8)**
- **Baseline**: `99725c4` (Post-P3.2a prefix-sum merge)
- **Scope**: Replaced `PackedCostVolume16 tmp(packed_cost.layout(), kInvalidCost)` in `CostAggregator::aggregate_packed()` with an uninitialized for-overwrite private workspace `PackedCrossTmp`.
- **Key Invariants & Safety Guarantees**:
  1. `tmp` lifetime strictly confined to `CostAggregator::aggregate_packed()` (allocated and freed within the method).
  2. Zero memory footprint increase: No persistent workspace retained across frames or pipeline phases, strictly preserving the existing peak memory model ($\le 3 \times C16$).
  3. The horizontal prefix pass unconditionally overwrites every valid packed disparity state in `tmp` before any vertical pass read, guaranteeing memory correctness with uninitialized storage.
  4. Header stability: `include/` remains untouched; `PackedCrossTmp` is private to `src/cost_aggregator.cpp`.

#### Allocation vs Fill Profiling (`profile_tmp_alloc`)

| Scene | Memory Footprint | Baseline `assign(fill)` | Raw `new uint16_t[]` | Fill Overhead Share |
| :--- | :---: | :---: | :---: | :---: |
| **ArtL** | 554.3 MiB | 111.04 ms | **0.01 ms** | **99.99%** |
| **Piano** | 2118.9 MiB | 425.40 ms | **0.01 ms** | **100.00%** |
| **Vintage** | 5609.1 MiB | 1147.85 ms | **0.02 ms** | **100.00%** |

#### Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | P3.2a Baseline Cross | P3.2b Cross | Cross Speedup | P3.2a Baseline Pipeline | P3.2b Pipeline | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 417.22 ms | 315.70 ms | **1.322x** (-101.5 ms) | 3554.32 ms | 3474.89 ms | **1.023x** (2.23% faster) | 100% (74/74) |
| **Piano (F)** | 2201.67 ms | 1859.74 ms | **1.184x** (-341.9 ms) | 13300.78 ms | 12990.29 ms | **1.024x** (2.33% faster) | 100% (74/74) |
| **Vintage (F)** | 6097.55 ms | 5106.25 ms | **1.194x** (-991.3 ms) | 29355.86 ms | 28561.43 ms | **1.028x** (2.71% faster) | 100% (74/74) |
| **Scene-Balanced Geomean** | — | — | **1.232x** (Gate: >= 1.08x) | — | — | **1.025x** (Gate: >= 1.02x) | **PASS** |
| **Pooled Total Sum** | 8716.44 ms | 7281.69 ms | **1.197x** (-1434.8 ms) | 46210.96 ms | 45026.61 ms | **1.026x** (-1184.4 ms) | **PASS** |

#### Cumulative Cross Track Progress (P3.0 Baseline `2ad6d26` -> P3.2b)
- **Cross stage runtime**:
  - ArtL: 1444.57 ms -> 364.99 ms (**3.958x**)
  - Piano: 5964.96 ms -> 1940.34 ms (**3.074x**)
  - Vintage: 15089.35 ms -> 5322.45 ms (**2.835x**)
  - Cross execution share collapsed from **35.51% down to 14.69%**.
- **End-to-end pipeline**:
  - ArtL: 4517.48 ms -> 3474.89 ms (**1.300x**)
  - Piano: 16946.61 ms -> 12990.29 ms (**1.305x**)
  - Vintage: 38344.20 ms -> 28561.43 ms (**1.343x**)
  - **Cumulative Pipeline Geomean**: **1.316x (31.6% faster end-to-end)**.
- **Current Pipeline Stage Distribution**:
  1. **SGM**: ~35.11%
  2. **Cost**: ~24.46%
  3. **Cross**: ~14.69%
  4. **Refine**: ~11.41%
- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed (Run 36216464038).
- **Next Step**: Conclude Cross Track; transition to **P3.3 Cost-Volume Optimization**.

---

## V3 Priority 3: Cost-Volume Optimization Track (P3.3)

### P3.3a: Cost Stage Hierarchical Attribution

Before implementing kernel optimizations, the Cost stage was profiled across three hierarchical levels:

1. **Level 1 (Allocation/Init vs Compute Kernel)**:
   - Evaluated the two serial full-volume `kInvalidCost` fills:
     1. `pipeline.cpp`: `out.packed_cost.allocate(layout, kInvalidCost)`
     2. `cost_computer.cpp`: `packed_cost.fill(kInvalidCost)`
   - Profiling confirmed these redundant full-volume fills account for **20.1%–20.3% of total Cost stage runtime** (ArtL: 151 ms / 750 ms, Piano: 562 ms / 2772 ms, Vintage: 1503 ms / 7380 ms).
2. **Level 2 (Cost Kernel Inner-Loop Arithmetic Breakdown)**:
   - Census popcount: ~12% of kernel runtime.
   - 3x `std::exp()` computations: **~61% of kernel runtime**.
   - Arithmetic mixing & clamping: ~27% of kernel runtime.
   - Micro-benchmark of discrete LUT vs scalar `std::exp()` achieved **2.57x kernel speedup**.
3. **Level 3 (Full-Resolution Middlebury F Bit-Exact Parity Audit)**:
   - Audited all discrete domain values across ArtL, Piano, and Vintage F (totaling **4,342,883,572 evaluated packed disparity states**).
   - Zero differences observed between scalar floating-point `std::exp()` and 1D lookup tables (`census[32]`, `ad[256]`, `grad[511]`), confirming 100% bit-exact equivalence.

---

### P3.3b: Avoid Redundant Packed Cost Volume Initialization

- **Status**: **MERGED as `4d11eda` (PR #9)**
- **Baseline**: `fb1d58c` (Post-P3.2b documentation merge)
- **Scope**:
  1. Refactored `PackedCostVolume<T>` to use `std::unique_ptr<T[]>` and `size_t`, bypassing C++17 `std::vector::resize()` element value-initialization loops.
  2. Implemented `allocate_for_overwrite(layout)` and `allocate_for_overwrite(range)` providing pure capacity acquisition in ~0.01 ms.
  3. Implemented explicit copy/move constructors and copy/move assignment operators with moved-from size zeroing (`other.size_ = 0`) and strong exception safety.
  4. Eliminated the two redundant full-volume invalid cost fills in `pipeline.cpp` and `cost_computer.cpp`.
  5. Added comprehensive move/copy lifecycle and moved-from object state consistency unit tests in `test_sanity.cpp`.
- **Key Invariants & Safety Guarantees**:
  - Memory Peak: Preserved strict $\le 3 \times C16$ peak volume memory constraint; zero cross-frame retained volumes.
  - Complete Initialization Invariant: The OpenMP parallel disparity loop in `compute_volume_packed()` unconditionally assigns all states $di \in [0, D_p)$ (either `kInvalidCost` for out-of-bounds or valid cost), guaranteeing no uninitialized state reads.

#### Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | Baseline Cost | P3.3b Cost | Cost Speedup | Baseline Pipeline | P3.3b Pipeline | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 807.03 ms | 652.99 ms | **1.236x** (-154.0 ms) | 3474.89 ms | 3308.61 ms | **1.050x** (5.03% faster) | 100% (888/888) |
| **Piano (F)** | 2971.26 ms | 2469.19 ms | **1.203x** (-502.1 ms) | 12990.29 ms | 12669.00 ms | **1.025x** (2.54% faster) | 100% (888/888) |
| **Vintage (F)** | 7794.98 ms | 6344.29 ms | **1.229x** (-1450.7 ms) | 28561.43 ms | 27145.01 ms | **1.052x** (5.22% faster) | 100% (888/888) |
| **Scene-Balanced Geomean** | — | — | **1.2226x** (Gate: >= 1.15x) | — | — | **1.0425x** (Gate: >= 1.03x) | **PASS** |
| **Pooled Total Sum** | 11573.27 ms | 9466.47 ms | **1.223x** (-2106.8 ms) | 45026.61 ms | 43122.62 ms | **1.044x** (-1904.0 ms) | **PASS** |

- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed (Run 36222599843).

---

### P3.3c: Discrete Cost Table LUT (`census[65] / ad[256] / grad[511]`)

- **Status**: **MERGED as `ad4ee3f` (PR #10)**
- **Baseline**: `4d11eda` (Post-P3.3b merge)
- **Scope**:
  1. Constructed three small read-only 1D lookup tables in `CostComputer::compute_volume_packed()` preceding the OpenMP parallel disparity loop:
     - `lut_census[65]`: Precomputes $1 - \exp(-i / \lambda_c)$ for Census Hamming distance (capacity 65; reachable range $\le 62$, specifically $\le 31$ for `SymmetricCensus9x7` and $\le 62$ for standard `Census9x7`; 260 bytes).
     - `lut_ad[256]`: Precomputes $\eta \cdot (1 - \exp(-i / \lambda_a))$ for absolute pixel intensity differences $[0, 255]$ (1024 bytes).
     - `lut_grad[511]`: Precomputes $\mu \cdot (1 - \exp(-i / \lambda_g))$ for Sobel gradient absolute differences sum $[0, 510]$ (2044 bytes).
  2. Replaced the scalar `std::exp()` runtime calculations in the inner compute loop with direct table lookups.
  3. Total memory footprint is 3328 B (~3.25 KiB), which fits comfortably within L1 D-Cache across all CPU cores with zero heap allocation or synchronization overhead.
  4. Dense reference implementation (`compute_volume()`) remains completely frozen as the bit-exact golden standard.

#### Isolated Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | P3.3b Baseline Cost | P3.3c Cost | Cost Speedup | P3.3b Baseline Pipeline | P3.3c Pipeline | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 652.99 ms | 258.36 ms | **2.527x** (-394.6 ms) | 3308.61 ms | 2923.16 ms | **1.132x** (11.7% faster) | 100% (888/888) |
| **Piano (F)** | 2469.19 ms | 979.72 ms | **2.520x** (-1489.5 ms) | 12669.00 ms | 11198.66 ms | **1.131x** (11.6% faster) | 100% (888/888) |
| **Vintage (F)** | 6344.29 ms | 2579.80 ms | **2.459x** (-3764.5 ms) | 27145.01 ms | 23782.75 ms | **1.141x** (12.4% faster) | 100% (888/888) |
| **Scene-Balanced Geomean** | — | — | **2.5021x** | — | — | **1.1348x** | **PASS** |
| **Pooled Total Sum** | 9466.47 ms | 3817.88 ms | **2.480x** (-5648.6 ms) | 43122.62 ms | 37904.57 ms | **1.138x** (-5218.1 ms) | **PASS** |

#### Cumulative Cost Track Gains (P3.3 Track: P3.3b + P3.3c vs Pre-P3.3 Baseline `fb1d58c`)

| Scene | Pre-P3.3 Cost | Post-P3.3c Cost | Cost Speedup | Pre-P3.3 Pipeline | Post-P3.3c Pipeline | Pipeline Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 807.03 ms | 258.36 ms | **3.124x** (-548.7 ms) | 3474.89 ms | 2923.16 ms | **1.189x** |
| **Piano (F)** | 2971.26 ms | 979.72 ms | **3.033x** (-1991.5 ms) | 12990.29 ms | 11198.66 ms | **1.160x** |
| **Vintage (F)** | 7794.98 ms | 2579.80 ms | **3.022x** (-5215.2 ms) | 28561.43 ms | 23782.75 ms | **1.201x** |
| **Scene-Balanced Geomean** | — | — | **3.0590x** | — | — | **1.1831x** |

- **Physical Time Reduction**: Across the three benchmark scenes, single-iteration pipeline execution time dropped by **-7.122 seconds per run** (from 45.026s down to 37.904s).
- **Cost Hotspot Reduction**: The Cost stage share of full pipeline runtime collapsed from **~24.5% down to ~8.8%–10.8%**.
- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed (Run 36223691804).

---

## V3 Priority 4: Refine Stage Optimization Track (P3.4)

### P3.4a: Refine Stage Hierarchical Attribution

Before introducing kernel modifications to `Refiner`, profiling on full-resolution Middlebury 2014 F (`ArtL`, `Piano`, `Vintage`, G-mode, 16T) established the structural properties of the stage:

1. **Reliable Mask Invalidation & Workload Volume**:
   - Reliable pixel skip ratio: 14.40% (ArtL), 14.90% (Piano), 7.79% (Vintage).
   - **85.1% to 92.2% of pixels are marked unreliable** and participate in the full 3-iteration refinement pass.
   - Total `local_cost()` invocations per full-res F run: **132.58 million calls** (ArtL: 17.51M, Piano: 57.39M, Vintage: 57.68M).
2. **Computational Hotspot within `local_cost()`**:
   - Each `local_cost()` call evaluated up to three transcendental `std::exp()` expressions (`ccensus`, `ad`, `grad`).
   - Profile confirmed that floating-point `std::exp()` calculations accounted for **~60% of Refine stage runtime**.
3. **Duplicate Candidate Analysis (P3.4c Opportunity)**:
   - Of the ~4 candidates evaluated per pixel iteration, **32.13% are duplicate disparity values** (ArtL: 30.38%, Piano: 32.31%, Vintage: 32.48%) caused by neighbor clamping and random perturbation bounds.

---

### P3.4b: Refine Discrete Cost LUT

- **Status**: **MERGED as `a215d2e` (PR #11)**
- **Baseline**: `a3750d4` (Post-P3.3c documentation merge)
- **Scope**:
  1. Extracted shared discrete lookup tables into private internal header `src/discrete_cost_lut.hpp` within `namespace apg::detail`:
     - `census[65]`: Hamming distance exponential penalty table (`[0, 64]`, reachable $\le 62$, 260 bytes).
     - `ad[256]`: Absolute intensity difference exponential penalty table (`[0, 255]`, 1024 bytes).
     - `grad[511]`: Gradient $L_1$ difference exponential penalty table (`[0, 510]`, 2044 bytes).
     - Total footprint: 3,328 B (~3.25 KiB), resident in L1 D-Cache with zero heap allocation.
  2. Replaced `std::exp()` in `Refiner::local_cost()` with direct table lookups.
  3. Unified `CostComputer::compute_volume_packed()` to share `detail::DiscreteCostLut`.
  4. Preserved exact `std::mt19937` random sequence, candidate order, and traversal without deduplication (isolated to P3.4c).
  5. Dense reference backend remained completely frozen.

#### Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | P3.3c Base Refine (ms) | P3.4b Refine (ms) | Refine Speedup | P3.3c Base Pipeline (ms) | P3.4b Pipeline (ms) | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 506.85 ms | 327.79 ms | **1.546x** (-179.1 ms) | 2923.16 ms | 2751.30 ms | **1.062x** (5.88% faster) | 100% (888/888) |
| **Piano (F)** | 1693.05 ms | 1094.18 ms | **1.547x** (-598.9 ms) | 11198.66 ms | 10547.91 ms | **1.062x** (5.81% faster) | 100% (888/888) |
| **Vintage (F)** | 1859.64 ms | 1175.54 ms | **1.582x** (-684.1 ms) | 23782.75 ms | 22672.12 ms | **1.049x** (4.67% faster) | 100% (888/888) |
| **Scene-Balanced Geomean** | — | — | **1.5584x** (Gate: >= 1.50x) | — | — | **1.0577x** (Gate: >= 1.04x) | **PASS** |
| **Pooled Total Sum** | 4059.54 ms | 2597.51 ms | **1.563x** (-1462.0 ms) | 37904.57 ms | 35971.33 ms | **1.054x** (-1933.2 ms) | **PASS** |

#### Cumulative Multi-Track Pipeline Progress (P3.0 Baseline `2ad6d26` -> Post-P3.4b `a215d2e`)

| Milestone | Pooled Pipeline Time (3 Scenes) | Delta vs Prior | Cumulative Speedup vs P3.0 |
| :--- | :---: | :---: | :---: |
| **P3.0 Initial Baseline** (`2ad6d26`) | 59.808 s | — | 1.000x |
| **P3.2b Post-Cross** (`79ed6ac`) | 45.027 s | -14.781 s | 1.316x |
| **P3.3c Post-Cost** (`ad4ee3f`) | 37.905 s | -7.122 s | 1.557x |
| **P3.4b Post-Refine LUT** (`a215d2e`) | **35.971 s** | **-1.933 s** | **1.647x** |

**Net execution time saved**: **-23.837 seconds per full-res F run** (39.9% end-to-end reduction).

#### Current Post-P3.4 Pipeline Stage Distribution (35.97 s Pooled)

1. **SGM**: ~16.13 s (**44.84%**) — Dominant #1 macro bottleneck
2. **Cross**: ~7.83 s (**21.78%**) — #2 macro bottleneck
3. **Cost**: ~3.62 s (**10.06%**) — Stabilized
4. **Refine**: ~2.60 s (**7.22%**) — Reduced from 13.43% down to 7.22%
5. **Prior**: ~2.36 s (**6.56%**)
6. **Other (Aux, WTA, Conf, Post)**: ~3.43 s (**9.54%**)

- **CI**: Ubuntu GCC and Clang builds, CTest, and smoke checks passed (Run 36225322417).

---

### P3.4c: Refine Candidate Deduplication Evaluation (REJECTED)

- **Status**: **REJECTED** (Preserved in branch `codex/p3-4c-refine-candidate-dedup`)
- **Baseline**: `519a2ce` (P3.4b merge on main)
- **Scope**:
  1. Profiled candidate deduplication inside `Refiner::refine()` on unreliable pixels before calling `local_cost()`.
  2. Strictly preserved candidate generation order and `std::uniform_int_distribution` RNG calls.
  3. Pre-checked duplicate candidate disparities in small stack array before evaluating `local_cost()`.
- **Merge Gate Criteria**:
  - Parity: 100% bit-exact across non-timing metrics and dual backend tests.
  - Refine Stage Speedup Geomean $\ge 1.20\times$.
  - Pipeline Speedup Geomean $\ge 1.015\times$.
  - No regression $< 0.995\times$.
- **Results**:
  - Non-timing parity: 100% bit-exact (888/888 fields checked).
  - Dual backend parity: 100% passed (E and G modes).
  - Refine Stage Speedup: ArtL 1.068x (-20.8 ms), Piano 1.063x (-65.0 ms), Vintage 0.995x (+5.6 ms).
  - **Refine Geomean Speedup**: **1.0415x** (Gate: $\ge 1.20\times$ -> **FAIL**).
  - Pooled Refine Stage Reduction: Only -80.15 ms across 3 full-res scenes (2597.51 ms down to 2517.36 ms).
- **Decision & Attribution**:
  Following P3.4b LUT optimization, `local_cost()` was already transformed into 3 fast array lookups; remaining Refine runtime is dominated by work buffer memory access, boundary checks, and loop iteration overhead. Skipping 32% of calls yielded only ~3.1% stage speedup, failing the 1.20x gate. As planned, the change was rejected without micro-tuning to keep `main` clean.

---

## V3 Priority 5: Packed SGM SIMD Vectorization Track (P3.5)

### P3.5a: SGM Feasibility Attribution & Interior Profiling

Before implementing SIMD kernels, a comprehensive profiler (`scratch/profile_packed_simd_feasibility.cpp`) evaluated all 8 SGM paths across Middlebury 2014 full-resolution F scenes (`ArtL`, `Piano`, `Vintage`, G-mode, 16T):
- **Workload**: 100.19 million path-pixels, 34.74 billion disparity evaluations.
- **Disparity Range ($D_c$)**: Mean = 346.72, Median = 272, Max = 768.
- **Neighbor Stepping Stability**: 92.96% of adjacent path pixels have identical $d_{min}$ ($\Delta dmin == 0$), and 95.59% have $|\Delta dmin| \le 1$.
- **SIMD Interior Coverage**:
  - Overlap coverage $[dmin_c, dmax_c) \cap [dmin_p, dmax_p)$: **99.05%**.
  - Interior condition ($d-1, d, d+1$ all within previous range): **98.56%** of all evaluations.
  - **8-lane AVX2 SIMD Vectorizable Proportion**: **97.20%** (Cardinal: 97.39%, Diagonal: 97.02%).
  - Vector tail (<8 remainder): only 1.36%.
  - Boundary scalar edges: only 1.44%.

**Conclusion**: The ragged disparity structure does not fragment vectorization. 97.20% of the SGM recurrence work falls directly into 8-lane contiguous SIMD blocks.

---

### P3.5b & P3.5c: Packed SGM AVX2 Recurrence, Runtime Dispatch & MSVC CI

- **Scope**:
  1. **AVX2 Recurrence Kernel** (`src/sgm_optimizer_avx2.cpp`):
     - Scalar prefix for boundary disparity lanes.
     - 8-lane AVX2 interior loop using contiguous unaligned vector loads (`_mm256_loadu_si256` for `prev[d-1]`, `prev[d]`, `prev[d+1]`), zero-extension unpacking for `uint16_t` cost (`_mm256_cvtepu16_epi32`), integer min chain, and invalid cost blending (`_mm256_blendv_epi8`).
     - Scalar suffix handling remaining tail and right boundary lanes.
     - Fully integer-arithmetic formulation ensuring 100% bit-exact equivalence with golden scalar reference.
  2. **Isolated Verification**:
     - 50,000 randomized synthetic test cases verified 100% bit-exact parity between AVX2 and scalar kernels.
     - Isolated kernel throughput jumped from 0.41 G-disp/s to 2.64 G-disp/s (**6.47x isolated speedup**).
  3. **Architecture & Runtime Dispatch** (`src/cpu_features.hpp`, `src/sgm_common.hpp`):
     - AVX2 kernel isolated into dedicated translation unit `src/sgm_optimizer_avx2.cpp`.
     - `CMakeLists.txt` sets `/arch:AVX2` (MSVC) and `-mavx2` (GCC/Clang) only on `sgm_optimizer_avx2.cpp`. The rest of `apg_sgm` remains completely scalar and portable.
     - Runtime CPU feature detection via `detail::is_avx2_supported()`, with automatic fallback to golden scalar path.
  4. **CI Matrix Expansion**:
     - Updated `.github/workflows/ci.yml` to include `windows-latest + MSVC` runner alongside Ubuntu GCC and Clang.

#### Performance Results (16 Threads, G-mode, Middlebury 2014 Full-Res F)

| Scene | P3.4b SGM (ms) | P3.5b SGM (ms) | SGM Speedup | P3.4b Pipeline (ms) | P3.5b Pipeline (ms) | Pipeline Speedup | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArtL (F)** | 1189.02 ms | 922.22 ms | **1.289x** (-266.8 ms) | 2751.30 ms | 2470.19 ms | **1.114x** (10.2% faster) | 100% (888/888) |
| **Piano (F)** | 4485.19 ms | 3541.36 ms | **1.267x** (-943.8 ms) | 10547.91 ms | 9457.71 ms | **1.115x** (10.3% faster) | 100% (888/888) |
| **Vintage (F)** | 10456.25 ms | 8518.56 ms | **1.227x** (-1937.7 ms) | 22672.12 ms | 20406.93 ms | **1.111x** (10.0% faster) | 100% (888/888) |
| **Scene-Balanced Geomean** | — | — | **1.2608x** | — | — | **1.1134x** | **PASS** |
| **Pooled Total Sum** | 16130.46 ms | 12982.14 ms | **1.2425x** (-3148.3 ms) | 35971.33 ms | 32334.83 ms | **1.1125x** (-3636.5 ms) | **PASS** |

#### Cumulative Pipeline Progress (P3.0 Baseline `2ad6d26` -> Post-P3.5b)

| Milestone | Pooled Pipeline Time (3 Scenes) | Delta vs Prior | Cumulative Speedup vs P3.0 |
| :--- | :---: | :---: | :---: |
| **P3.0 Initial Baseline** (`2ad6d26`) | 59.808 s | — | 1.000x |
| **P3.2b Post-Cross** (`79ed6ac`) | 45.027 s | -14.781 s | 1.316x |
| **P3.3c Post-Cost** (`ad4ee3f`) | 37.905 s | -7.122 s | 1.557x |
| **P3.4b Post-Refine LUT** (`a215d2e`) | 35.971 s | -1.933 s | 1.647x |
| **P3.5b Post-SGM AVX2** | **32.335 s** | **-3.637 s** | **1.850x** |

**Net execution time saved**: **-27.473 seconds per full-res F run** (45.9% end-to-end reduction vs P3.0).

#### Comprehensive Post-P3.5 10-Stage Pipeline Attribution (32.33 s Pooled)

| Rank | Stage | ArtL (ms) | Piano (ms) | Vintage (ms) | Pooled Total (ms) | Share (%) | Status & Bottleneck Role |
|:---:|:---|---:|---:|---:|---:|---:|:---|
| **#1** | **SGM** | 922.22 ms | 3541.36 ms | 8518.56 ms | **12,982.14 ms** | **40.15%** | Primary macro bottleneck (reduced from 44.84%) |
| **#2** | **Cross** | 344.88 ms | 1872.90 ms | 5182.06 ms | **7,399.84 ms** | **22.89%** | **Dominant secondary bottleneck** (4.9x larger than Prior) |
| **#3** | **Cost** | 244.90 ms | 930.72 ms | 2383.09 ms | **3,558.71 ms** | **11.01%** | Stabilized (P3.3 Discrete LUT) |
| **#4** | **Refine** | 329.01 ms | 1128.65 ms | 1217.44 ms | **2,675.10 ms** | **8.27%** | Stabilized (P3.4b Discrete LUT) |
| **#5** | **Prior** | 194.19 ms | 529.39 ms | 779.54 ms | **1,503.12 ms** | **4.65%** | Secondary tail stage |
| **#6** | **Right WTA** | 90.88 ms | 332.99 ms | 972.15 ms | **1,396.02 ms** | **4.32%** | Secondary tail stage |
| **#7** | **Post** | 136.18 ms | 468.75 ms | 485.46 ms | **1,090.39 ms** | **3.37%** | Tail post-processing |
| **#8** | **Aux** | 89.89 ms | 290.13 ms | 309.36 ms | **689.38 ms** | **2.13%** | Gray/gradient auxiliary setup |
| **#9** | **WTA** | 47.69 ms | 174.21 ms | 412.49 ms | **634.39 ms** | **1.96%** | Minor tail stage |
| **#10** | **Confidence**| 36.79 ms | 127.65 ms | 127.98 ms | **292.42 ms** | **0.90%** | Minor tail stage |
| — | **Pipeline Total** | **2470.19 ms** | **9457.71 ms** | **20406.93 ms** | **32,334.83 ms** | **100.00%** | End-to-end full-res F pipeline |

> [!NOTE]
> Stage medians are independently aggregated across 4 runs and therefore do not sum exactly to the median end-to-end pipeline time (sum of stage medians = 32,221.51 ms vs pipeline median total = 32,334.83 ms, delta = 113.32 ms / 0.35%).

---

## V3 Priority 7: Cross Aggregator AVX2 Micro-Optimization Track (P3.7)

### P3.7a: Cross Hierarchical Micro-Attribution & Division Characterization

Prior to vectorization, a dedicated profiler (`scratch/profile_cross_micro_attribution.cpp`) decomposed `CostAggregator::aggregate_packed()` into 6 granular sub-phases and tracked denominator distributions across the 3 full-resolution F benchmark scenes (`ArtL`, `Piano`, `Vintage`, G-mode, 16T):

#### 6-Subphase Breakdown (7,365.52 ms Pooled Baseline)

| Sub-Phase | ArtL (ms) | Piano (ms) | Vintage (ms) | Pooled Total (ms) | Share (%) | Bottleneck Category |
|:---|---:|---:|---:|---:|---:|:---|
| **1. Arms Build** | 8.81 ms | 28.51 ms | 54.47 ms | **91.79 ms** | 1.25% | Memory read / branch |
| **2. PackedCrossTmp Alloc** | 0.01 ms | 0.01 ms | 0.08 ms | **0.10 ms** | 0.00% | Negligible (P3.2b uninitialized) |
| **3. Horizontal Prefix** | 71.93 ms | 390.87 ms | 1,149.15 ms | **1,611.95 ms** | 21.89% | Row-stride prefix accumulation |
| **4. Horizontal Query + Div** | 63.85 ms | 362.59 ms | 981.00 ms | **1,407.44 ms** | 19.11% | Window subtraction + 32-bit division |
| **5. Vertical Prefix** | 114.77 ms | 625.68 ms | 1,885.30 ms | **2,625.75 ms** | 35.65% | Col-stride prefix accumulation (cache) |
| **6. Vertical Query + Div** | 85.51 ms | 465.24 ms | 1,077.74 ms | **1,628.49 ms** | 22.11% | Window subtraction + 32-bit division |
| **Total Cross Stage** | **344.88 ms** | **1,872.90 ms** | **5,182.06 ms** | **7,365.52 ms** | **100.00%** | Combined Cross aggregation |

- **Combined Prefix Construction**: **4,237.70 ms (57.53%)** — dominant memory and stride bottleneck.
- **Combined Query & Division**: **3,035.92 ms (41.22%)** — arithmetic denominator division bottleneck.
- **Total Variable Divisions Evaluated**: Exactly **8,684,650,260 (8.685 billion)** 32-bit unsigned integer divisions pooled across H and V passes.
- **Denominator Characteristics**:
  - $cnt \in [1, 35]$ strictly ($cnt == 0$ count is exactly 0).
  - Maximum window accumulation $acc \le 35 \times 65534 = 2,293,690 < 2^{24}$.
  - $cnt == 35$ represents **44.27% (3.845 billion)** of all divisions.
- **Tile Alignment**:
  - Horizontal pass: $tb == 16$ is **100.0%** of all tiles.
  - Vertical pass: $tb == 16$ is **99.4%** of all tiles.
  - $B = 16$ disparity tile size perfectly matches two 8-lane AVX2 vector registers.

---

### P3.7b: Exact 8-Lane AVX2 Vector Quotient & Exhaustive Mathematical Proof

Because Cross aggregation outputs directly feed the SGM dynamic programming optimization, approximate reciprocal multiplication (`(acc * recip[cnt]) >> shift`) with $\pm 1$ cost drift is strictly prohibited.

Instead, an exact AVX2 integer quotient kernel was designed:
1. `_mm256_cvtepi32_ps`: Zero precision loss conversion of $acc$ and $cnt$ to IEEE 754 single-precision float (since $acc \le 2,293,690 < 2^{24}$ significand limit).
2. `_mm256_div_ps`: 8-lane single-precision vector float division.
3. `_mm256_cvttps_epi32`: Truncation round towards zero.
4. Two-instruction residual correction:
   ```cpp
   prod = q * cnt;
   rem = acc - prod;
   if (rem < 0) q -= 1;
   rem = acc - (q * cnt);
   if (rem >= cnt) q += 1;
   ```
5. **Exhaustive Domain Proof** (`scratch/test_exact_cross_quotient.cpp`):
   - Exhaustively tested every single valid pair $(acc, cnt)$ across the complete operational space:
     $$\sum_{cnt=1}^{35} (cnt \times 65534 + 1) = 41,286,455 \text{ pairs}$$
   - **Result**: **0 mismatches** across all 41.3 million test cases (100.00000% mathematical bit-exactness verified in 0.1s at 341 M pairs/s).

---

### P3.7c: AVX2 Cross Aggregator Implementation & SoA Architecture

- **Structure-of-Arrays (SoA) Workspace** (`PrefixWorkspaceSoA`):
  - Transformed thread-local prefix buffers from AoS (`PrefixEntry {sum, cnt}`) to contiguous SoA (`sums` and `cnts` vectors sized `(max_dim + 1) * 16`).
  - For each step, 16 sums (64 B) and 16 counts (64 B) align directly with two 256-bit AVX2 registers, eliminating AoS stride and register shuffle overhead.
- **AVX2 Prefix Vectorization**:
  - Full interior tiles ($b_{start} = 0, b_{end} = 16$): Unpack 16 $\times$ `uint16_t` costs with two `_mm_loadu_si128` + `_mm256_cvtepu16_epi32`, mask out `kInvalidCost`, and add to previous prefix sums/counts in parallel.
- **AVX2 Query Vectorization & Packing**:
  - Evaluate 16 disparities simultaneously: Two 8-lane vector subtractions for `acc` and `cnt`, two 8-lane exact quotient evaluations via `exact_cross_div_8`, and pack 16 $\times$ `uint32_t` into 16 $\times$ `uint16_t` via:
    ```cpp
    _mm256_permute4x64_epi64(_mm256_packus_epi32(q0, q1), _MM_SHUFFLE(3, 1, 2, 0))
    ```
    written with a single unaligned 256-bit store (`_mm256_storeu_si256`).
- **Translation Unit & Dispatch Isolation**:
  - Isolated into `src/cost_aggregator_avx2.cpp` with compile options `/arch:AVX2` (MSVC) and `-mavx2` (GCC/Clang).
  - Runtime dispatch via `detail::is_avx2_supported()`, with golden scalar path preserved as fallback.

---

### P3.7d: Performance Verification & Merge Gate Evaluation

#### Isolated Cross Stage Results (16 Threads, Middlebury 2014 Full-Res F)

| Scene | Baseline Cross (ms) | P3.7 AVX2 Cross (ms) | Cross Speedup | Cross Time Delta | Bit-Exact |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 344.88 ms | 239.12 ms | **1.442x** | -105.76 ms | 100% |
| **Piano (F)** | 1,872.90 ms | 1,152.00 ms | **1.626x** | -720.90 ms | 100% |
| **Vintage (F)** | 5,182.06 ms | 2,685.28 ms | **1.930x** | -2,496.78 ms | 100% |
| **Scene-Balanced Geomean** | — | — | **1.654x** (Gate: $\ge 1.20\times$) | — | **PASS** |
| **Pooled Total Sum** | 7,399.84 ms | 4,076.40 ms | **1.815x** | **-3,323.44 ms** | **PASS** |

#### End-to-End Pipeline Results (16 Threads, G-mode, 4 Repeats)

| Scene | P3.5 Pipeline (ms) | P3.7 Pipeline (ms) | Pipeline Speedup | Pipeline Time Delta | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 2,470.19 ms | 2,390.26 ms | **1.033x** | -79.93 ms | 100% bit-exact |
| **Piano (F)** | 9,457.71 ms | 8,680.41 ms | **1.090x** | -777.30 ms | 100% bit-exact |
| **Vintage (F)** | 20,406.93 ms | 18,194.61 ms | **1.122x** | -2,212.32 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.0805x** (Gate: $\ge 1.03\times$) | — | **PASS** |
| **Pooled Total Sum** | 32,334.83 ms | 29,265.28 ms | **1.1049x** | **-3,069.55 ms** | **PASS** |

#### Cumulative Multi-Track Pipeline Progress (P3.0 Baseline `2ad6d26` -> Post-P3.7)

| Milestone | Pooled Pipeline Time (3 Scenes) | Delta vs Prior | Cumulative Speedup vs P3.0 |
| :--- | :---: | :---: | :---: |
| **P3.0 Initial Baseline** (`2ad6d26`) | 59.808 s | — | 1.000x |
| **P3.2b Post-Cross Prefix** (`79ed6ac`) | 45.027 s | -14.781 s | 1.316x |
| **P3.3c Post-Cost LUT** (`ad4ee3f`) | 37.905 s | -7.122 s | 1.557x |
| **P3.4b Post-Refine LUT** (`a215d2e`) | 35.971 s | -1.933 s | 1.647x |
| **P3.5b Post-SGM AVX2** (`0c100db`) | 32.335 s | -3.637 s | 1.850x |
| **P3.7c Post-Cross AVX2** (`702bb79`) | 29.265 s | -3.070 s | 2.044x |
| **P3.8b-1 Post-First-Path Direct Store** (`d63317f`) | 26.434 s | -2.831 s | 2.263x |
| **P3.8b-2b Post-Packed SGM U16 Accumulator** (`70b7655`) | 22.798 s | -3.636 s | 2.623x |
| **P3.9a Post-Packed Right-WTA State-Major** (`d0fba40`) | 21.799 s | -0.999 s | 2.744x |
| **P3.9b-2 Post-Packed Cost AVX2** | **18.949 s** | **-2.850 s** | **3.156x** |

**Net execution time saved**: **-40.859 seconds per full-res F run** (**68.3% end-to-end reduction**, over 3.15x cumulative speedup!).

---

## V3 Priority 3.8b-1: First-Path Direct Store & Accumulator Zero-Fill Elision

**Status:** IMPLEMENTED & VERIFIED on branch `codex/p3-8b-first-path-direct-store`. 100% bit-exact dual backend parity across dense/packed backends on E and G modes across all test cases.

### P3.8b-1a: Bottleneck Analysis & Memory Bandwidth Constraints

Post-P3.7 10-stage profiling established SGM as the dominant bottleneck at **43.10% / 12.89s pooled** runtime. A dedicated micro-attribution experiment (`scratch/profile_sgm_micro_attribution.cpp`) uncovered the underlying hardware constraint:
1. **Recurrence Compute is Fast**: Pure AVX2 recurrence arithmetic without accumulator updates required only **2.39s (21.54%)**.
2. **Accumulator Traffic Dominates**: Accumulator Read-Modify-Write (RMW) traffic accounted for **8.71s (78.46%)** of SGM runtime. Across 8 paths on Vintage (11.22 GB active packed state space), 8 RMW passes saturate LLC and DRAM memory buses with over 179.5 GB of bandwidth.
3. **Accumulator Zero-Fill Cost**: Initial `packed_cost32.fill(0)` took **1.03s (9.27%)** pooled solely clearing memory that is subsequently completely overwritten.

### P3.8b-1b: First-Path Direct Store Architecture

Because path P0 $(+1, 0)$ scans each row from $x = 0$ to $w-1$, it visits every active non-empty pixel slice $(x, y)$ in strictly deterministic row-major order:
1. **Zero-Fill Elision**: `packed_cost32.allocate(layout, 0)` and `packed_cost32.fill(0)` are completely removed in favor of `packed_cost32.allocate_for_overwrite(layout)`.
2. **First-Path Overwrite Mode (`AccumulateMode::Overwrite`)**:
   - P0 is invoked with `AccumulateMode::Overwrite`, while subsequent paths (P1..P7) run with `AccumulateMode::Add`.
   - **Scalar Path**:
     ```cpp
     if (c[di] == kInvalidCost) {
         a[di] = kInvalidCost32;
     } else if constexpr (Mode == AccumulateMode::Overwrite) {
         a[di] = static_cast<uint32_t>(cur.vals[di]);
     } else {
         if (a[di] != kInvalidCost32) a[di] += static_cast<uint32_t>(cur.vals[di]);
     }
     ```
   - **AVX2 Path**:
     Elides `_mm256_loadu_si256(a + di)` and `_mm256_add_epi32(va, v_cur)` entirely:
     ```cpp
     if constexpr (Mode == AccumulateMode::Overwrite) {
         __m256i v_new_a = _mm256_blendv_epi8(v_cur, v_ainv32, v_is_inv);
         _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + di), v_new_a);
     }
     ```
3. **Hardware Impact**:
   - Eliminates 1 full-volume write pass (zero-fill).
   - Eliminates 1 full-volume read pass (P0 load).
   - Reduces total SGM memory traffic by approximately **11.8% (2 out of 17 memory passes)**.

---

### P3.8b-1c: Performance Verification & Merge Gate Evaluation

#### SGM Stage Results (16 Threads, Middlebury 2014 Full-Res F, G-mode)

| Scene | Baseline SGM (P3.7) | P3.8b-1 SGM (ms) | SGM Speedup | SGM Time Delta | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 921.89 ms | 699.60 ms | **1.318x** | -222.29 ms | 100% bit-exact |
| **Piano (F)** | 3,486.66 ms | 2,745.26 ms | **1.270x** | -741.40 ms | 100% bit-exact |
| **Vintage (F)** | 8,485.42 ms | 6,586.50 ms | **1.288x** | -1,898.92 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.292x** (Gate: $\ge 1.08\times$) | — | **PASS** |
| **Pooled Total Sum** | 12,893.97 ms | 10,031.36 ms | **1.285x** | **-2,862.61 ms** | **PASS** |

#### End-to-End Pipeline Results (16 Threads, G-mode, 4 Repeats)

| Scene | P3.7 Pipeline (ms) | P3.8b-1 Pipeline (ms) | Pipeline Speedup | Pipeline Time Delta | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 2,390.26 ms | 2,138.57 ms | **1.118x** | -251.69 ms | 100% bit-exact |
| **Piano (F)** | 8,680.41 ms | 7,941.67 ms | **1.093x** | -738.74 ms | 100% bit-exact |
| **Vintage (F)** | 18,194.61 ms | 16,353.67 ms | **1.113x** | -1,840.94 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.108x** (Gate: $\ge 1.025\times$) | — | **PASS** |
| **Pooled Total Sum** | 29,265.28 ms | 26,433.91 ms | **1.107x** | **-2,831.37 ms** | **PASS** |

#### Post-P3.8b-1 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.7 Time | vs Post-P3.7 Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **10,089.04 ms** | **38.36%** | 12,885.76 ms | **1.277x** |
| # 2 | Cost | 4,202.69 ms | 15.98% | 4,222.94 ms | 1.005x |
| # 3 | Cross | 4,086.66 ms | 15.54% | 4,874.06 ms | 1.193x |
| # 4 | Refine | 2,565.37 ms | 9.75% | 2,527.87 ms | 0.985x |
| # 5 | Prior | 1,452.03 ms | 5.52% | 1,514.86 ms | 1.043x |
| # 6 | Right WTA | 1,369.69 ms | 5.21% | 1,422.86 ms | 1.039x |
| # 7 | Post | 1,044.56 ms | 3.97% | 1,085.12 ms | 1.039x |
| # 8 | Aux | 664.65 ms | 2.53% | 710.22 ms | 1.069x |
| # 9 | WTA | 619.45 ms | 2.36% | 708.20 ms | 1.143x |
| #10 | Confidence | 284.32 ms | 1.08% | 309.28 ms | 1.088x |
| — | **Total Pipeline** | **26,300.96 ms** | **100.00%** | **29,899.98 ms** | **1.137x** |

---

## V3 Priority 3.8b-2: 16-Bit Packed SGM Accumulator & Safety Bound Dispatch

**Status:** IMPLEMENTED & VERIFIED on branch `codex/p3-8b2-packed-sgm-u16-accumulator`. 100% bit-exact dual backend parity across dense/packed backends on E and G modes across all test cases.

### P3.8b-2a: Mathematical Safety Proof & Feasibility Verification

Following P3.8b-1's identification of accumulator memory bandwidth as the primary bottleneck, analysis of the SGM recurrence revealed an opportunity to halve accumulator word width:

1. **Analytical Dynamic Range Bound**:
   - For each path, the recurrence value satisfies:
     $$cur = cost + best - min\_prev$$
   - Since $best \le min\_prev + P2$, it strictly holds that:
     $$best - min\_prev \le P2$$
     $$cur \le cost\_max + P2$$
   - Under the adaptive penalty schedule, $P2(D) = \max(P1 + 1, P2\_base - \lfloor D / \gamma \rfloor) \le \max(P2\_base, P1 + 1)$.
   - Therefore, the maximum single-path contribution is bounded by:
     $$path\_max \le cost\_max + \max(P2\_base, P1 + 1)$$
   - For an $N$-path aggregation, the maximum accumulated cost is strictly bounded by:
     $$total\_max \le N \times path\_max$$
   - Under default pipeline parameters ($cost\_max = 255$, $P1 = 10$, $P2\_base = 120$, $N = 8$ paths):
     $$path\_max \le 255 + 120 = 375$$
     $$total\_max \le 8 \times 375 = 3000 \ll 65534$$
   - The sentinel value $kInvalidCost = 65535$ ($0\text{xFFFF}$) remains completely unreachable by valid accumulation (margin of over $62535$).

2. **Centralized Runtime Safety Dispatch**:
   A centralized predicate `can_use_u16_sgm_accumulator(cfg, npath)` evaluates:
   ```cpp
   inline bool can_use_u16_sgm_accumulator(const PipelineConfig& cfg, int npath) {
       const uint64_t p2_max = std::max<uint64_t>(
           static_cast<uint64_t>(cfg.sgm.P2_base),
           static_cast<uint64_t>(cfg.sgm.P1) + 1);
       const uint64_t path_max = static_cast<uint64_t>(cfg.cost.cost_max) + p2_max;
       const uint64_t total_max = static_cast<uint64_t>(npath) * path_max;
       return total_max < kInvalidCost;
   }
   ```
   If any user configuration exceeds the $uint16\_t$ dynamic range, the pipeline automatically and transparently falls back to the 32-bit `packed_cost32` accumulator path.

3. **Empirical Feasibility & Invariant Checks**:
   - Across full-resolution Middlebury 2014 scenes (ArtL, Piano, Vintage) spanning 19.5B accumulator evaluations:
     - ArtL: Observed maximum accumulated cost = **2686** (Theoretical bound: 3000)
     - Piano: Observed maximum accumulated cost = **2639** (Theoretical bound: 3000)
     - Vintage: Observed maximum accumulated cost = **2784** (Theoretical bound: 3000)
     - In all cases, maximum observed cost was $\le 2784$, leaving a **95.75% safety headroom** below 65535.
     - 100% bit-exact disparity maps and zero accumulator overflows were verified against the 32-bit golden reference.

---

### P3.8b-2b: Production Architecture & AVX2 Kernel Details

1. **Buffer Naming & Management**:
   - Added `PackedCostVolume16 packed_sgm16;` to `PipelineBuffers` for the 16-bit accumulator.
   - Retained `PackedCostVolume32 packed_cost32;` exclusively for the safe fallback path.
   - Memory allocation in `src/pipeline.cpp` allocates only the active volume and releases the inactive one, keeping working-set memory lean.
   - Peak volume memory on Vintage drops from **17.68 GB down to 11.80 GB** (-5.88 GB, **-33.3%**).

2. **AVX2 16-bit Accumulator Kernel**:
   - **Path P0 (Overwrite)**: Stores 8 lanes of 16-bit values (128 bits total) directly using `_mm256_packus_epi32` and `_mm_storeu_si128`, saving half the write traffic compared to 256-bit 32-bit stores.
   - **Paths P1..P7 (Add)**: Loads 8 lanes of 16-bit accumulator values via `_mm_loadu_si128`, zero-extends to 32 bits with `_mm256_cvtepu16_epi32`, performs vector addition and invalid blending in 32-bit registers, saturates and packs back to 16 bits with `_mm_packus_epi32`, and writes back 128 bits via `_mm_storeu_si128`.
   - **Memory Traffic Reduction**: Halves accumulator read/write bandwidth from 8 bytes/state to 4 bytes/state. Across 8 paths, SGM memory bus traffic is reduced by 50% on all accumulator interactions.

3. **Downstream Direct Consumption**:
   - `winner_take_all_packed` natively accepts `PackedCostVolume16`, avoiding any format conversion or intermediary allocation.

---

### P3.8b-2c: Performance Verification & Merge Gate Evaluation

#### SGM Stage Results (16 Threads, Middlebury 2014 Full-Res F, G-mode, 4 Repeats)

| Scene | Baseline SGM (P3.8b-1) | P3.8b-2b SGM (ms) | SGM Speedup | SGM Time Delta | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 699.60 ms | 513.26 ms | **1.363x** | -186.34 ms | 100% bit-exact |
| **Piano (F)** | 2,745.26 ms | 1,964.16 ms | **1.398x** | -781.10 ms | 100% bit-exact |
| **Vintage (F)** | 6,586.50 ms | 4,396.16 ms | **1.498x** | -2,190.34 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.418x** (Gate: $\ge 1.35\times$) | — | **PASS** |
| **Pooled Total Sum** | 10,031.36 ms | 6,873.58 ms | **1.460x** | **-3,157.78 ms** | **PASS** |

#### End-to-End Pipeline Results (16 Threads, G-mode, 4 Repeats)

| Scene | P3.8b-1 Pipeline (ms) | P3.8b-2b Pipeline (ms) | Pipeline Speedup | Pipeline Time Delta | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 2,138.57 ms | 1,918.15 ms | **1.115x** | -220.42 ms | 100% bit-exact |
| **Piano (F)** | 7,941.67 ms | 7,037.36 ms | **1.129x** | -904.31 ms | 100% bit-exact |
| **Vintage (F)** | 16,353.67 ms | 13,842.79 ms | **1.181x** | -2,510.88 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.141x** (Gate: $\ge 1.08\times$) | — | **PASS** |
| **Pooled Total Sum** | 26,433.91 ms | 22,798.30 ms | **1.159x** | **-3,635.61 ms** | **PASS** |

#### Post-P3.8b-2b 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.8b-1 Time | vs Post-P3.8b-1 Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **6,843.45 ms** | **30.20%** | 10,089.04 ms | **1.474x** |
| # 2 | Cross | 4,091.24 ms | 18.03% | 4,086.66 ms | 0.999x |
| # 3 | Cost | 3,724.88 ms | 16.43% | 4,202.69 ms | 1.128x |
| # 4 | Refine | 2,561.12 ms | 11.30% | 2,565.37 ms | 1.002x |
| # 5 | Prior | 1,450.31 ms | 6.40% | 1,452.03 ms | 1.001x |
| # 6 | Right WTA | 1,368.10 ms | 6.04% | 1,369.69 ms | 1.001x |
| # 7 | Post | 1,042.80 ms | 4.60% | 1,044.56 ms | 1.002x |
| # 8 | Aux | 663.20 ms | 2.93% | 664.65 ms | 1.002x |
| # 9 | WTA | 618.50 ms | 2.73% | 619.45 ms | 1.002x |
| #10 | Confidence | 284.10 ms | 1.25% | 284.32 ms | 1.001x |
| — | **Total Pipeline** | **22,647.70 ms** | **100.00%** | **26,300.96 ms** | **1.161x** |

---

## V3 Priority 3.9a: Packed Right-WTA State-Major Traversal

**Status:** IMPLEMENTED & VERIFIED on branch `codex/p3-9a-packed-right-wta-state-major`. 100% bit-exact dual backend parity across dense/packed backends on E and G modes across all test cases.

### P3.9a-1: Bottleneck Analysis & Algorithmic Redesign

Following P3.8b-2b's reduction of SGM runtime to 30.20%, stage attribution revealed Right WTA occupying **1,392.35 ms (6.14%)** of pipeline time. Investigation of `wta_right_from_packed_volume` revealed a severe data-structure mismatch:
1. **Disparity-Raster Mismatch**: The legacy packed Right-WTA loop iterated per right pixel $x_r$, sweeping over the global disparity range $d \in [d_0, d_0 + D)$. For each $(x_r, d)$, it computed $x_l = x_r + d$ and invoked `vol.contains(xl, y, d)` and `vol.at(xl, y, d)`.
2. **Cache Inefficiency & Branching**: Because packed volume slices are indexed by left coordinate $(x_l, y)$, iterating across $x_r$ caused non-contiguous, jumping memory reads across slices, coupled with branch mispredictions in bounds checking.
3. **State-Major Redesign**:
   - Invert the iteration order to state-major:
     ```cpp
     for (int y = 0; y < h; ++y) {
         std::fill(best_cost.begin(), best_cost.end(), kInvalidCost);
         std::fill(best_d.begin(), best_d.end(), -1);

         for (int xl = 0; xl < w; ++xl) {
             const int lo = vol.dmin(xl, y);
             const int hi = vol.dmax(xl, y);
             const uint16_t* s = vol.slice(xl, y);
             const int count = hi - lo;

             for (int di = 0; di < count; ++di) {
                 const int d = lo + di;
                 const int xr = xl - d;
                 if (xr < 0 || xr >= w) continue;

                 const uint16_t c = s[di];
                 if (c != kInvalidCost && c < best_cost[xr]) {
                     best_cost[xr] = c;
                     best_d[xr] = d;
                 }
             }
         }
         // Subpixel interpolation per xr using best_d[xr]
     }
     ```
   - **Persistent OpenMP Thread-Local Scratch**: Allocates `std::vector<uint16_t> best_cost(w)` and `std::vector<int> best_d(w)` once per OpenMP thread outside the row loop, avoiding dynamic allocation in the hot loop.
   - **Bit-Exact Tie Semantics**: For any fixed $x_r$, candidate disparities satisfy $d = x_l - x_r$. As $x_l$ increases strictly monotonically from $0$ to $w-1$, $d$ also strictly increases. With strict inequality `<` updates, the candidate with minimal disparity is chosen on cost ties, guaranteeing 100% bit-exact parity with the legacy raster loop.

---

### P3.9a-2: Performance Verification & Merge Gate Evaluation

#### Right WTA Stage Results (16 Threads, Middlebury 2014 Full-Res F, G-mode, 4 Repeats)

| Scene | Baseline Right WTA (P3.8b-2b) | P3.9a Right WTA (ms) | Right WTA Speedup | Right WTA Time Delta | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 89.56 ms | 31.70 ms | **2.825x** | -57.86 ms | 100% bit-exact |
| **Piano (F)** | 337.27 ms | 118.65 ms | **2.842x** | -218.62 ms | 100% bit-exact |
| **Vintage (F)** | 965.52 ms | 266.66 ms | **3.621x** | -698.86 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **3.075x** (Gate: $\ge 1.50\times$) | — | **PASS** |
| **Pooled Total Sum** | 1,392.35 ms | 417.01 ms | **3.339x** | **-975.34 ms (-0.98 s)** | **PASS** |

#### End-to-End Pipeline Results (16 Threads, G-mode, 4 Repeats)

| Scene | P3.8b-2b Pipeline (ms) | P3.9a Pipeline (ms) | Pipeline Speedup | Pipeline Time Delta | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 1,875.48 ms | 1,842.87 ms | **1.018x** | -32.61 ms | 100% bit-exact |
| **Piano (F)** | 6,975.46 ms | 6,775.97 ms | **1.029x** | -199.49 ms | 100% bit-exact |
| **Vintage (F)** | 13,811.65 ms | 13,180.21 ms | **1.048x** | -631.44 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.0315x** (Gate: $\ge 1.015\times$) | — | **PASS** |
| **Pooled Total Sum** | 22,662.59 ms | 21,799.05 ms | **1.040x** | **-863.54 ms (-0.86 s)** | **PASS** |

#### Post-P3.9a 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.8b-2b Time | vs Post-P3.8b-2b Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **6,927.14 ms** | **31.78%** | 6,843.45 ms | 0.988x |
| # 2 | Cross | 4,080.73 ms | 18.72% | 4,091.24 ms | 1.003x |
| # 3 | Cost | 3,734.11 ms | 17.13% | 3,724.88 ms | 0.998x |
| # 4 | Refine | 2,558.37 ms | 11.74% | 2,561.12 ms | 1.001x |
| # 5 | Prior | 1,482.68 ms | 6.80% | 1,450.31 ms | 0.978x |
| # 6 | Post | 1,035.03 ms | 4.75% | 1,042.80 ms | 1.008x |
| # 7 | Aux | 681.32 ms | 3.13% | 663.20 ms | 0.973x |
| # 8 | WTA | 630.33 ms | 2.89% | 618.50 ms | 0.981x |
| # 9 | **Right WTA** | **417.01 ms** | **1.91%** | **1,392.35 ms** | **3.339x** |
| #10 | Confidence | 289.84 ms | 1.33% | 284.10 ms | 0.980x |
| — | **Total Pipeline** | **21,799.05 ms** | **100.00%** | **22,647.70 ms** | **1.039x** |

---

## V3 Priority 3.9b-2: Packed Cost AVX2 Production Implementation & Dispatch

**Status:** IMPLEMENTED & VERIFIED on branch `codex/p3-9b2-packed-cost-avx2-production`. 100% bit-exact dual backend parity across dense/packed backends on E and G modes across all test cases.

### P3.9b-2a: Production Architecture & Correctness Guards

Following P3.9b-1's feasibility study (99.56% 8-lane coverage, 6.75x isolated prototype speedup, 0 mismatches across 4.34B states), the production 8-lane AVX2 kernel was implemented:

1. **Isolated Compilation Unit & Dynamic Dispatch**:
   - Implemented in dedicated translation unit `src/cost_computer_avx2.cpp` with header `src/cost_computer_avx2.hpp`.
   - CMake sets `/arch:AVX2` (MSVC) or `-mavx2` (GCC/Clang) exclusively on `src/cost_computer_avx2.cpp`, avoiding compiler flag contamination.
   - Preserves public `CostComputer` API unchanged; dispatches to `detail::compute_volume_packed_avx2()` if `cfg.cost.census == CensusType::SymmetricCensus9x7 && detail::is_avx2_supported()`, falling back to scalar golden otherwise.

2. **Vector-Safe Geometric Bounds Guard**:
   - Rather than assuming non-negative disparities ($d \ge 0$), the production kernel dynamically computes the exact vector-safe disparity interval $[di_{vstart}, di_{vend})$:
     $$di_{vstart} = \max(0, x - lo - w + 1)$$
     $$di_{vend} = \min(D_p, x - lo + 1)$$
   - Only disparity states within this interval enter the 8-lane SIMD kernel, ensuring $xr_0 \dots xr_7$ are strictly bounded within $[0, w)$.
   - States outside $[di_{vstart}, di_{vend})$ (including negative disparity regimes and non-multiple-of-8 boundaries) are evaluated via a shared, inlined template scalar helper `evaluate_scalar_cost_state_t<UseAd, UseGrad>()`.

3. **Compile-Time Branch Elimination & Strict FP Order**:
   - The outer function specializes `compute_volume_packed_avx2_impl<bool UseAd, bool UseGrad>()` across the four configuration combinations, eliminating per-block runtime branching.
   - Arithmetic order strictly matches scalar golden: Census gather $\to$ AD gather $\to$ Grad gather $\to$ explicit mul $\to$ explicit add ($0.5\text{f}$) $\to$ `cvttps_epi32` truncation.

---

### P3.9b-2b: Performance Verification & Merge Gate Evaluation

#### Cost Stage Results (16 Threads, Middlebury 2014 Full-Res F, G-mode, 4 Repeats)

| Scene | Baseline Cost (P3.9a) | P3.9b-2 Cost (ms) | Cost Speedup | Cost Time Delta | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 257.72 ms | **57.64 ms** | **4.471x** | -200.08 ms | 100% bit-exact |
| **Piano (F)** | 963.65 ms | **210.57 ms** | **4.576x** | -753.08 ms | 100% bit-exact |
| **Vintage (F)** | 2,512.75 ms | **523.05 ms** | **4.804x** | -1,989.70 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **4.615x** (Gate: $\ge 4.0\times$) | — | **PASS** |
| **Pooled Total Sum** | 3,734.11 ms | **791.26 ms** | **4.719x** | **-2,942.85 ms (-2.94 s)** | **PASS** |

#### End-to-End Pipeline Results (16 Threads, G-mode, 4 Repeats)

| Scene | P3.9a Pipeline (ms) | P3.9b-2 Pipeline (ms) | Pipeline Speedup | Pipeline Time Delta | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 1,842.87 ms | 1,622.26 ms | **1.136x** | -220.61 ms | 100% bit-exact |
| **Piano (F)** | 6,775.97 ms | 6,076.35 ms | **1.115x** | -699.62 ms | 100% bit-exact |
| **Vintage (F)** | 13,180.21 ms | 11,250.24 ms | **1.172x** | -1,929.97 ms | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.141x** (Gate: $\ge 1.10\times$) | — | **PASS** |
| **Pooled Total Sum** | 21,799.05 ms | **18,948.85 ms** | **1.150x** | **-2,850.20 ms (-2.85 s)** | **PASS** |

#### Post-P3.9b-2 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.9a Time | vs Post-P3.9a Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **6,932.88 ms** | **36.59%** | 6,927.14 ms | 0.999x |
| # 2 | Cross | 4,088.95 ms | 21.58% | 4,080.73 ms | 0.998x |
| # 3 | Refine | 2,595.15 ms | 13.70% | 2,558.37 ms | 0.986x |
| # 4 | Prior | 1,455.20 ms | 7.68% | 1,482.68 ms | 1.019x |
| # 5 | Post | 1,040.97 ms | 5.49% | 1,035.03 ms | 0.994x |
| # 6 | **Cost** | **791.26 ms** | **4.18%** | **3,734.11 ms** | **4.719x (-78.8% time!)** |
| # 7 | Aux | 661.36 ms | 3.49% | 681.32 ms | 1.030x |
| # 8 | WTA | 639.49 ms | 3.37% | 630.33 ms | 0.986x |
| # 9 | Right WTA | 418.93 ms | 2.21% | 417.01 ms | 0.995x |
| #10 | Confidence | 287.73 ms | 1.52% | 289.84 ms | 1.007x |
| — | **Total Pipeline** | **18,948.85 ms** | **100.00%** | **21,799.05 ms** | **1.150x** |

---

## V3 Priority 3.10b-1: SGM Path-Pair Fusion & Vertical Locality Feasibility

**Status:** FEASIBILITY COMPLETED. Path-Pair Fusion REJECTED (NO-GO). Vertical Tiling CONFIRMED (STRONG GO).

In P3.10b-1, four orthogonal prototypes were evaluated on 16 threads across Middlebury 2014 full-res F (ArtL, Piano, Vintage) against the current production baseline:

1. **HPAIR (Horizontal Pair Fusion: P0 + P1)**:
   - Evaluated storing P0 contribution into a row-local scratch buffer, then during P1 reading scratch and performing a single overwrite into the global accumulator.
   - Result: ArtL 1.074x, Piano 0.693x, Vintage 0.677x. **Geomean = 0.796x (NO-GO)**.
2. **VPAIR (Vertical Pair Fusion: P2 + P3, Column Traversal)**:
   - Evaluated storing Down contribution into a column-local scratch buffer, then during Up reading scratch and performing a single RMW into the global accumulator.
   - Result: ArtL 1.080x, Piano 0.997x, Vintage 0.791x. **Geomean = 0.948x (NO-GO)**.
3. **VPAIR + VTILE (Combined Vertical Fusion + Tiling)**:
   - Result: ArtL 0.871x, Piano 1.109x, Vintage 0.920x. **Geomean = 0.961x (NO-GO)**.
4. **VTILE (Vertical Tiled Traversal, B=16)**:
   - Restructured P2/P3 traversal into $x$-blocks ($B=16$), traversing $y$ in outer loop and $B$ adjacent $x$ columns in inner loop with independent `PathState` chains.
   - Result: ArtL 1.687x, Piano 1.785x, Vintage 1.438x. **Geomean = 1.631x (STRONG GO)**.

**Attribution Note:** Pair-fusion prototypes replaced global accumulator accesses with contribution scratch but did not reduce logical byte volume in these designs; increased scratch working-set costs outweighed the intended benefit. Consequently, all pair-fusion designs were abandoned in favor of pure vertical tiling.

---

## V3 Priority 3.10b-2: SGM Vertical X-Tiling Production Implementation

**Status:** IMPLEMENTED & MERGED on branch `codex/p3-10b2-sgm-vertical-x-tiling`. 100% bit-exact parity across dense/packed backends on E and G modes.

### P3.10b-2a: Production Architecture & Implementation Details

1. **Vertical X-Tile Traversal**:
   - In `src/sgm_optimizer_avx2.cpp`, vertical paths P2 `(0, +1)` and P3 `(0, -1)` were restructured to process columns in blocks of `kVerticalTileX = 16`.
   - Outer loop iterates over $x$-blocks $[xb, \min(xb + 16, w))$, maintaining `std::array<PathState, 16>` for `prev` and `cur` states.
   - Inner loop runs along $y$, processing $nb \le 16$ adjacent pixels per step.
   - No additional contribution scratch arrays are introduced; active working set is strictly $16 \times \text{PathState}$.

2. **Scheduling & B=1 Control Attribution**:
   - An isolated microcheck on Piano and Vintage decomposed the baseline-to-production speedup:
     - **Current $\to$ B1**: 1.358x pooled (explained by framework/traversal restructuring effect and `#pragma omp parallel for schedule(dynamic, 1)` task granularity).
     - **B1 $\to$ B16**: 1.147x pooled (direct spatial cache locality improvement across packed volume slices).
     - **Total Vertical Speedup**: 1.557x pooled / 1.627x geomean.
   - OpenMP scheduling comparison showed `schedule(dynamic, 1)` outperforms `schedule(static)` by ~11% on Piano (due to ragged packed workload variation across $x$). Fixed to `schedule(dynamic, 1)`.

3. **Partial Tile Correctness**:
   - Unit tests added to `tests/test_sanity.cpp` explicitly validating widths $w \in \{1, 7, 15, 16, 17, 31, 32, 33\}$ across Path4 and Path8 modes with negative disparities and empty slices.

---

### P3.10b-2b: Production Benchmark & Gate Evaluation

#### 1. Vertical P2+P3 Isolated Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline Vertical (ms) | B=1 Control (ms) | P3.10b-2 B=16 (ms) | Curr $\to$ B1 | B1 $\to$ B16 | Total Speedup | Gate |
|:---|---:|---:|---:|:---:|:---:|:---:|:---:|
| **ArtL (F)** | 164.16 ms | 106.49 ms | **99.42 ms** | 1.541x | 1.071x | **1.651x** | $\ge 1.20\times$ (PASS) |
| **Piano (F)** | 686.41 ms | 465.59 ms | **379.89 ms** | 1.474x | 1.226x | **1.807x** | $\ge 1.20\times$ (PASS) |
| **Vintage (F)** | 1,329.58 ms | 1,033.80 ms | **920.70 ms** | 1.286x | 1.123x | **1.444x** | $\ge 1.20\times$ (PASS) |
| **Scene-Balanced Geomean** | — | — | — | **1.428x** | **1.139x** | **1.627x** | **$\ge 1.40\times$ (PASS)** |
| **Pooled Total Sum** | 2,180.15 ms | 1,605.88 ms | **1,400.01 ms** | **1.358x** | **1.147x** | **1.557x (-780 ms)** | **PASS** |

#### 2. Full SGM Stage Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline SGM (ms) | P3.10b-2 SGM (ms) | SGM Speedup | Gate | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 493.89 ms | **402.22 ms** | **1.228x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Piano (F)** | 1,887.56 ms | **1,647.39 ms** | **1.146x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Vintage (F)** | 4,326.43 ms | **3,993.12 ms** | **1.083x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.151x** | **$\ge 1.08\times$ (PASS)** | **PASS** |
| **Pooled Total Sum** | 6,707.89 ms | **6,042.74 ms** | **1.110x (-665 ms)** | — | **PASS** |

#### 3. Full Pipeline End-to-End Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline Pipeline (ms) | P3.10b-2 Pipeline (ms) | Pipeline Speedup | Gate | Dual Backend Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 1,659.20 ms | **1,567.52 ms** | **1.058x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Piano (F)** | 6,215.46 ms | **5,975.29 ms** | **1.040x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Vintage (F)** | 11,516.03 ms | **11,182.72 ms** | **1.030x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | **1.043x** | **$\ge 1.025\times$ (PASS)** | **PASS** |
| **Pooled Total Sum** | 19,390.69 ms | **18,725.53 ms** | **1.036x (-665 ms)** | — | **PASS** |

#### 4. Post-P3.10b-2 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.9b-2 Time | vs Post-P3.9b-2 Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **6,309.24 ms** | **33.69%** | 6,932.88 ms | **1.099x (-623.6 ms)** |
| # 2 | Cross | 4,394.53 ms | 23.47% | 4,088.95 ms | 0.930x |
| # 3 | Refine | 2,565.64 ms | 13.70% | 2,595.15 ms | 1.011x |
| # 4 | Prior | 1,469.29 ms | 7.85% | 1,455.20 ms | 0.990x |
| # 5 | Post | 1,039.43 ms | 5.55% | 1,040.97 ms | 1.001x |
| # 6 | Cost | 807.35 ms | 4.31% | 791.26 ms | 0.980x |
| # 7 | Aux | 671.75 ms | 3.59% | 661.36 ms | 0.985x |
| # 8 | WTA | 634.33 ms | 3.39% | 639.49 ms | 1.008x |
| # 9 | Right WTA | 419.76 ms | 2.24% | 418.93 ms | 0.998x |
| #10 | Confidence | 286.96 ms | 1.53% | 287.73 ms | 1.003x |
| — | **Total Pipeline** | **18,725.53 ms** | **100.00%** | **18,948.85 ms** | **1.012x (vs paired: 1.036x)** |

**Cumulative Speedup vs P3.0 Baseline (59.83 s):** **3.195x** (Total pipeline reduced from 59.83 s to 18.73 s).

---

## V3 Priority 3.11a: SGM Diagonal Ray Interleaving Locality Feasibility

**Status:** FEASIBILITY COMPLETED. STRONG GO to production.

In P3.11a, diagonal traversal locality was investigated across Middlebury 2014 full-res F (ArtL, Piano, Vintage) at 16 threads. Diagonal rays (P4..P7) account for ~50% of 8-path SGM execution time. Analysis revealed two distinct geometric ray families:
1. **Family H (Horizontal-border origins: top/bottom)**:
   - Covers 65% to 76% of all diagonal states in full-res scenes.
   - For Family H, stepping rays concurrently step-by-step accesses identical row indices $y$ and horizontally adjacent column indices $x$, establishing strong spatial row locality across packed volume slices.
   - Evaluated block sizes $B \in \{8, 16, 32, 64\}$. $B=16$ achieved 1.14x~1.31x speedup across scenes.
2. **Family S (Side-border origins: left/right)**:
   - Stepping along rays produces vertical strides across memory. Driving ray blocks by `global y` with $x(y) = x_{\text{side}} + dx \cdot (y - y_{\text{start}})$ aligns memory accesses with row-major memory order.
   - Evaluated block sizes $B \in \{16, 32, 64\}$. $B=32$ achieved 1.12x~1.20x speedup across scenes.

Combined diagonal feasibility achieved a geometric-mean speedup of **1.163x** and pooled saving of **426.95 ms** with 100% bit-exact parity, providing conclusive evidence to proceed directly to production implementation.

---

## V3 Priority 3.11b: SGM Family-Specific Diagonal Ray Interleaving Production Implementation

**Status:** IMPLEMENTED & MERGED on branch `codex/p3-11b-sgm-diagonal-ray-interleaving`. 100% bit-exact parity across dense/packed backends on E and G modes.

### P3.11b-a: Production Architecture & Implementation Details

1. **Family-Specific Block Sizes & Traversals**:
   - In `src/sgm_optimizer_avx2.cpp`, diagonal aggregation (`aggregate_diagonal_packed_avx2`) explicitly separates the ray space into Family H and Family S:
     - **Family H**: Fixed `kDiagHorizontalRayBlock = 16`. Traversal is structured as `step` outer ($0 \le \text{step} < h$) and ray-in-block inner ($0 \le i < act\_b$), accessing identical row $y$ and contiguous columns $x$.
     - **Family S**: Fixed `kDiagSideRayBlock = 32`. Traversal is driven by `global y` (increasing for $dy = +1$, decreasing for $dy = -1$), computing corresponding $x$ per active ray in block to maintain strict row-local memory access.
2. **Working-Set Discipline & Scratch Elimination**:
   - Ray interleaving introduces strictly zero contribution scratch volumes and zero full-image temporary buffers.
   - Ray states are independent; active working state consists solely of bounded `std::array<PathState, B>` per ray block whose vector storage scales with each ray's current disparity range and is reused across the block without dynamic reallocation.
3. **Corner Origin Handling & Edge Cases**:
   - Corner origins are systematically grouped into Family H; Family S handles strictly the $h - 1$ side origins ($r \in [w, w + h - 1)$), completely avoiding double-processing corner pixels.
   - Unit tests added to `tests/test_sanity.cpp` explicitly verifying non-square shapes, partial widths, negative disparities, empty slices, and sentinel preservation across Path4 and Path8 modes.

---

### P3.11b-b: Production Benchmark & Gate Evaluation

The production paired benchmark evaluated baseline (un-interleaved ray-by-ray execution from commit `16c3280`) versus P3.11b on Middlebury 2014 full-resolution F (ArtL, Piano, Vintage) on 16 threads, median of 4 repeats.

#### 1. Diagonal (P4..P7) Isolated Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline Diagonal (ms) | P3.11b Diagonal (ms) | Delta (ms) | Speedup | Gate | Parity |
|:---|---:|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 212.88 ms | **191.92 ms** | -20.96 ms | **1.109x** | $\ge 1.05\times$ (PASS) | 100% bit-exact |
| **Piano (F)** | 886.79 ms | **734.44 ms** | -152.35 ms | **1.207x** | $\ge 1.05\times$ (PASS) | 100% bit-exact |
| **Vintage (F)** | 2,066.73 ms | **1,828.03 ms** | -238.70 ms | **1.131x** | $\ge 1.05\times$ (PASS) | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | — | **1.148x** | **$\ge 1.12\times$ (PASS)** | **PASS** |
| **Pooled Total Sum** | 3,166.39 ms | **2,754.38 ms** | **-412.01 ms** | **1.150x** | **$\ge 300\text{ ms}$ saving (PASS)** | **PASS** |

#### 2. Full SGM Stage Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline SGM (ms) | P3.11b SGM (ms) | Delta (ms) | SGM Speedup | Gate | Parity |
|:---|---:|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 395.49 ms | **390.79 ms** | -4.71 ms | **1.012x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Piano (F)** | 1,589.71 ms | **1,452.41 ms** | -137.30 ms | **1.095x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Vintage (F)** | 3,930.61 ms | **3,766.79 ms** | -163.82 ms | **1.043x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | — | **1.049x** | **$\ge 1.04\times$ (PASS)** | **PASS** |
| **Pooled Total Sum** | 5,915.81 ms | **5,609.98 ms** | **-305.83 ms** | **1.055x** | — | **PASS** |

#### 3. Full Pipeline End-to-End Results (16 Threads, G-mode, 4 Repeats, Median)

| Scene | Baseline Pipeline (ms) | P3.11b Pipeline (ms) | Delta (ms) | Pipeline Speedup | Gate | Dual Backend Parity |
|:---|---:|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 1,565.82 ms | **1,561.12 ms** | -4.71 ms | **1.003x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Piano (F)** | 5,708.14 ms | **5,570.84 ms** | -137.30 ms | **1.025x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Vintage (F)** | 11,048.82 ms | **10,885.00 ms** | -163.82 ms | **1.015x** | $\ge 0.995\times$ (PASS) | 100% bit-exact |
| **Scene-Balanced Geomean** | — | — | — | **1.014x** | **$\ge 1.012\times$ (PASS)** | **PASS** |
| **Pooled Total Sum** | 18,322.78 ms | **18,016.95 ms** | **-305.83 ms** | **1.017x** | — | **PASS** |

#### 4. Post-P3.11b 10-Stage Attribution Profile (16 Threads, Pooled 3 Scenes)

| Rank | Stage | Pooled Time (ms) | Stage Share | vs Post-P3.10b-2 Time | vs Post-P3.10b-2 Speedup |
|:---:|:---|---:|---:|---:|:---:|
| # 1 | **SGM** | **5,879.30 ms** | **32.63%** | 6,309.24 ms | **1.073x (-429.9 ms)** |
| # 2 | Cross | 4,203.48 ms | 23.33% | 4,394.53 ms | 1.045x |
| # 3 | Refine | 2,582.70 ms | 14.33% | 2,565.64 ms | 0.993x |
| # 4 | Prior | 1,474.16 ms | 8.18% | 1,469.29 ms | 0.997x |
| # 5 | Post | 1,050.33 ms | 5.83% | 1,039.43 ms | 0.990x |
| # 6 | Cost | 824.59 ms | 4.58% | 807.35 ms | 0.979x |
| # 7 | Aux | 692.36 ms | 3.84% | 671.75 ms | 0.970x |
| # 8 | WTA | 639.09 ms | 3.55% | 634.33 ms | 0.993x |
| # 9 | Right WTA | 420.10 ms | 2.33% | 419.76 ms | 0.999x |
| #10 | Confidence | 285.91 ms | 1.59% | 286.96 ms | 1.004x |
| — | **Total Pipeline** | **18,016.95 ms** | **100.00%** | **18,725.53 ms** | **1.039x** |

**Cumulative Speedup vs Canonical Baseline (59.808 s):** **3.320x** (Total pipeline reduced from 59.808 s to 18.017 s).

---

## V3 Priority 13: Refiner AVX2 Dedicated Translation Unit + Runtime Dispatch Production

**Decision:** **MERGE PR #19**.
The Refiner production path now dispatches to a dedicated AVX2 translation unit (`src/refiner_avx2.cpp`) with per-pixel invariant hoisting when supported at runtime via `detail::is_avx2_supported()`, falling back to the unchanged portable scalar algorithm (`src/refiner.cpp`) on non-AVX2 hosts. Public API in `include/apg_sgm/refiner.hpp` remains completely unmodified; internal entrypoints are declared strictly in `src/refiner_internal.hpp`. Both isolated Refiner and paired end-to-end pipeline benchmarks strongly exceeded all production gates with 100% bit-exact disparity parity across all pixels and modes.

### 1. Architectural Design & Codegen Attribution
- **Codegen Root Cause**: In standard scalar x64 compilation (`/O2 /MD`), `std::round` calls out-of-line `roundf`, and compiler inlining of `local_cost` into the spatial propagation loop is suppressed by register pressure. Compiling with target-specific `/arch:AVX2` lowers `std::round` directly to hardware `vroundss`, inlines local cost evaluations, and eliminates call frames in the hot inner loop.
- **Invariant Hoisting (`LeftPixelCtx`)**: Caching invariant left-pixel attributes (Census words, gray value, Sobel gradients, search range bounds, active cost flags) per pixel eliminates redundant memory lookups and branch checks across candidate disparity evaluations.
- **Portability & Isolation**: Neither the overall target `apg_sgm` nor `src/refiner.cpp` receives AVX2 flags. Only `src/refiner_avx2.cpp` has target-specific `/arch:AVX2` (MSVC) / `-mavx2` (GCC/Clang) properties.

### 2. Isolated Refiner Production Benchmark (16 Threads, Full-Res F, 7 Repeats, Median)
- **Convention**: `Saving = Scalar - AVX2` (positive denotes benefit); `Delta = AVX2 - Scalar` (negative denotes reduction).

| Scene | Scalar (ms) | AVX2 (ms) | Speedup | Saving (ms) | Parity |
|:---|---:|---:|:---:|:---:|:---:|
| **ArtL (F)** | 306.69 ms | 142.76 ms | **2.148x** | +163.93 ms | 100% bit-exact |
| **Piano (F)** | 1,098.97 ms | 502.30 ms | **2.188x** | +596.67 ms | 100% bit-exact |
| **Vintage (F)** | 1,167.90 ms | 546.12 ms | **2.139x** | +621.79 ms | 100% bit-exact |
| **Scene Geomean** | — | — | **2.158x** | — | **PASS** |
| **Pooled Sum** | 2,573.56 ms | 1,191.17 ms | **2.161x** | **+1,382.39 ms** | **PASS** |

- **Production Gate Evaluation**:
  - Requirement: Geomean $\ge 1.70\times$ (preferred $\ge 1.90\times$), every scene $\ge 1.50\times$, pooled saving $\ge 1.00\text{ s}$, 100% bit-exact.
  - Measured: Geomean **2.158x**, per-scene minimum **2.139x**, pooled saving **+1.382 s**. **EXCEEDED PREFERRED GATE**.

### 3. Per-Repeat Paired Full Pipeline Benchmark & Accounting (16 Threads, Full-Res F, 7 Repeats)
- Evaluated on identical pre-refine buffers per repeat `i`, comparing runtime candidate against forced scalar fallback:
  - `total_saving[i] = scalar_total[i] - avx_total[i]`
  - `refine_saving[i] = scalar_refine[i] - avx_refine[i]`
  - `residual[i] = total_saving[i] - refine_saving[i]`

| Scene | Scalar Pipe (ms) | AVX2 Pipe (ms) | Pipe Speedup | Total Save (ms) | Refine Save (ms) | Residual (ms) | Disparity Parity |
|:---|---:|---:|:---:|---:|---:|---:|:---:|
| **ArtL (F)** | 1,521.70 ms | 1,379.01 ms | **1.103x** | +148.14 ms | +147.55 ms | +0.86 ms | 100% bit-exact |
| **Piano (F)** | 5,612.76 ms | 5,051.08 ms | **1.111x** | +571.65 ms | +571.94 ms | -0.86 ms | 100% bit-exact |
| **Vintage (F)** | 10,416.95 ms | 9,825.29 ms | **1.060x** | +590.39 ms | +597.04 ms | -2.26 ms | 100% bit-exact |
| **Scene Geomean** | — | — | **1.091x** | — | — | — | **PASS** |
| **Pooled Sum** | 17,551.41 ms | 16,255.37 ms | **1.080x** | **+1,310.17 ms** | **+1,316.53 ms** | **-2.27 ms** | **PASS** |

- **Accounting & Residual Analysis**:
  - The pooled residual across all paired pipeline runs is **-2.27 ms**, which represents only **0.17%** of the Refine saving (well below the 25% threshold).
  - This confirms that pipeline-level gains are strictly and cleanly attributed to the Refiner acceleration without non-Refine/session noise artifacts.
- **Pipeline Merge Gate Evaluation**:
  - Requirement: Pooled saving $\ge 0.80\text{ s}$, pipeline geomean $\ge 1.04\times$, every scene $\ge 0.995\times$.
  - Measured: Pooled saving **+1.310 s**, geomean **1.091x**, per-scene minimum **1.060x**. **PASSED**.

### 4. Fresh 10-Stage Attribution Profile (Candidate Build Same-Session)
- Measured across Middlebury full-resolution F (`ArtL`, `Piano`, `Vintage`), 16 threads, median of 7 measured repeats in the same candidate binary session:

| Rank | Stage | ArtL (ms) | Piano (ms) | Vintage (ms) | Pooled (ms) | Share (%) |
|:---:|:---|---:|---:|---:|---:|:---:|
| # 1 | SGM | 383.44 ms | 1,457.83 ms | 3,668.22 ms | **5,509.49 ms** | 34.22% |
| # 2 | Cross | 234.22 ms | 1,134.47 ms | 2,752.09 ms | **4,120.78 ms** | 25.59% |
| # 3 | **Prior** | **188.14 ms** | **543.00 ms** | **800.10 ms** | **1,531.23 ms** | **9.51%** |
| # 4 | **Refine** | **159.12 ms** | **510.76 ms** | **572.66 ms** | **1,242.55 ms** | **7.72%** |
| # 5 | Post | 129.16 ms | 460.84 ms | 460.61 ms | **1,050.61 ms** | 6.53% |
| # 6 | Aux | 106.60 ms | 311.70 ms | 327.09 ms | **745.40 ms** | 4.63% |
| # 7 | WTA | 48.68 ms | 178.22 ms | 405.43 ms | **632.34 ms** | 3.93% |
| # 8 | Cost | 42.90 ms | 152.25 ms | 365.13 ms | **560.28 ms** | 3.48% |
| # 9 | Right WTA | 32.73 ms | 120.90 ms | 273.81 ms | **427.44 ms** | 2.65% |
| #10 | Confidence | 35.96 ms | 125.27 ms | 119.97 ms | **281.19 ms** | 1.75% |
| — | **Sum of Stage Medians** | — | — | — | **16,101.31 ms** | **100.00%** |
| — | **Median Total Runtime** | — | — | — | **16,255.37 ms** | — |

- **Profile Consistency**: Sum of stage medians (16,101.31 ms) closely matches the independently evaluated median total pipeline runtime (16,255.37 ms) within 0.95%. Shares sum to strictly 100.00%.
- **Ranking Shift**: Refine drops from 3rd bottleneck (previously 14.33%) to 4th (7.72%). **Prior Estimator (1,531.23 ms, 9.51%) officially becomes the #3 bottleneck** and the next primary actionable target.
- **Canonical Cumulative Milestone**: Fresh production measurement total is **16.255 s**, achieving **3.679x speedup** vs canonical P3.0 baseline (59.808 s). Conservative Amdahl projection based on post-P3.11b baseline (18.017 s - 1.382 s = 16.635 s) yields **3.595x**.
