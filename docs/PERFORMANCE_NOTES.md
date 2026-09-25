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

**Updated decision:** the B-only P2.1a candidate passes the agreed performance gate on the same-binary paired measurement and is ready for merge review. The earlier separate-build `0.99446x` total and Cross slowdown are retained as exploratory data but are not repeatable in the controlled follow-up. P2.1b remains rejected. P2.2 remains a separate task and was not started here.
