# Architecture — Adaptive Prior-Guided Census SGM

## Why not a stack

Putting full AD-Census, full SGM, full ELAS and full PatchMatch in series duplicates smoothness and explodes the parameter surface. Each algorithm donates only the piece that is uniquely good at its job.

```
Local Robust Matching          Census + AD + Gradient
        |
        v
Global Regularization          Adaptive SGM
        |                      (light Cross is a preconditioner only)
        v
Prior / Hard-region Refine     ELAS-like prior + selective PatchMatch
```

## Module map

```
include/apg_sgm/
  types.hpp                 QualityMode, PathType, CensusType
  config.hpp                PipelineConfig::from_mode()
  image.hpp                 Image8 / Image32f + PGM I/O
  buffers.hpp               CostVolume, SearchRange, PipelineBuffers
  cost_computer.hpp         descriptors + combined cost
  prior_estimator.hpp       sparse supports + d_prior + range
  cost_aggregator.hpp       Cross x1 (H then V)
  sgm_optimizer.hpp         4/8-path adaptive SGM + WTA + subpixel
  confidence_estimator.hpp  uniqueness / LR / texture
  refiner.hpp               neighbor + random hypothesis propagation
  postprocess.hpp           LR classify, fill, median
  pipeline.hpp              StereoMatcher facade
```

## Cost model

Symmetric Census 9x7 compares pairs across the center, 31 bits in uint32. Hamming via XOR + popcount.

```
C = 1 - exp(-C_census / lc)
  + eta (1 - exp(-C_AD / la))
  + mu  (1 - exp(-C_grad / lg))
```

then quantized to uint16 in [0, cost_max].

## Adaptive SGM P2

```
gradient < T1          P2 = P2_base
T1 <= gradient < T2    P2 = P2_base / 2
gradient >= T2         P2 = P2_base / 4
```

Always P2 >= P1 + 1.

## Prior (V2)

Not a full ELAS clone. Extract supports with C1 << C2 + texture + cheap LR, interpolate d_prior on a 16px grid, search d in [d_prior - R, d_prior + R].

Full Delaunay planes are the upgrade when roads/walls/floors dominate.

## Refinement (V3)

Not full-image PatchMatch. Uncertain pixels try {d, neighbors, d+rand} scored with local combined cost. 2-3 raster passes.

## Post-process

No Region Voting x5.

```
LR check
  occlusion  -> background-biased fill (smaller d)
  mismatch   -> color-similar neighbor, else d_prior, else interp
median
```
