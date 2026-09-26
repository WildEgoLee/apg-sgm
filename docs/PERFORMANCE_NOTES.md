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
