# Adaptive Prior-Guided Census SGM

Traditional (non-neural) stereo matching pipeline. Not a stack of full AD-Census + SGM + ELAS + PatchMatch. Each algorithm donates only the piece it is uniquely good at.

| Layer | Role | Source |
|------|------|------|
| Local Robust Matching | Census + AD + Gradient | AD-Census / libSGM Symmetric Census 9x7 |
| Cost preconditioner | one edge-aware Cross pass | AD-Census Cross, H+V once |
| Global Regularization | Adaptive P1/P2 4/8-path SGM | SGM |
| Prior | sparse supports -> d_prior -> shrink search range | ELAS prior only |
| Hard-region refinement | hypothesis propagation on low-confidence pixels | PatchMatch-inspired, no full planes |
| Post | LR, occlusion vs mismatch fill, median | simplified; no Region Voting x5 |

This repo ships a **runnable V1 core**. V2/V3 are feature-flagged so stages can be ablated.

## Modes

```
Fast        Sym-Census -> 4-path SGM -> WTA -> Subpixel -> LR
Balanced    Census+AD+Grad -> Cross x1 -> Adaptive 4-path SGM -> Conf -> LR/Fill
HighQuality Sparse prior -> Adaptive range -> 8-path SGM -> Conf -> selective refine -> fill
```

Default mode is **Balanced**.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires C++17. OpenMP optional. I/O is binary PGM/PPM only (no OpenCV).

## Run

Images must already be rectified.

```bash
./build/apg-stereo left.pgm right.pgm disp.pgm --mode balanced --disp-max 128
```

## Class flow

```
CostComputer -> PriorEstimator -> CostAggregator -> SgmOptimizer
    -> ConfidenceEstimator -> Refiner -> PostProcessor
```

See [docs/DATAFLOW.md](docs/DATAFLOW.md) for which buffers are W x H x D vs streamable, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## Development order

1. V1 (default path): combined cost + optional Cross x1 + adaptive 4-path SGM + WTA + subpixel + confidence + LR + fill + median
2. V2: enable `PriorParams.enable` (sparse support + range shrink)
3. V3: enable `RefineParams.enable` (uncertain-region hypothesis propagation; upgrade to planes later)

Do not turn on Voting, full PatchMatch planes, Cross x4 and full-image Delaunay at once.

## Memory

Dense cost volume is `uint16[W*H*D]`.

- 1280x720x128 ~ 225 MB
- 1920x1080x256 ~ 1.0 GB

V2 shrinks *compute* first. Packed `sum_p D(p)` storage comes after the prior is trusted.
