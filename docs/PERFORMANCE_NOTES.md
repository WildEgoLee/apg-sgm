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
- **Next Step**: Proceed to **P3.3c: Discrete Cost Table LUT** (`census[32] / ad[256] / grad[511]`).



