# Buffer and data-flow design

## Resident vs streamable

| Buffer | Shape | Notes |
|--------|-------|-------|
| left / right | W x H x C | rectified 8-bit |
| gray, gx, gy | W x H | 2D |
| census_left/right | W x H | streamable to compute; stored for reuse |
| d_prior | W x H | V2 |
| SearchRange | W x H | 2D |
| CostVolume cost | W x H x D | MUST be resident |
| disparity, confidence | W x H | after WTA |

Do not stream the cost volume. SGM paths need a full row/column of all d.

Layout: `index(x,y,d) = (y*W + x)*D + (d - d0)`

V1/V2 allocate global D. Tight SearchRange skips unused d (cost_max). Packed sum_p D(p) is a later optimization.

## Sequence

```
CostComputer.compute_aux          gray, grad, census     [2D]
PriorEstimator.estimate           d_prior, range         [2D, V2]
CostComputer.compute_volume       cost                   [3D]
CostAggregator.aggregate          in-place cost          [3D, optional]
SgmOptimizer.optimize             cost := sum L_r        [3D]
SgmOptimizer.winner_take_all      disparity              [2D]
right WTA from same volume        disparity_right        [2D]
ConfidenceEstimator               mask                   [2D]
Refiner.refine                    uncertain pixels       [V3]
PostProcessor                     LR, fill, median
```

Right disparity: C_R(xr, d) = C_L(xr+d, d). No second SGM in V1.
