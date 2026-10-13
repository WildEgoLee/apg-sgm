# APG-SGM P4 Architecture & Engineering Roadmap

## 1. Overview & Vision

APG-SGM P3 established a high-performance stereo matching engine on CPU (4.09x cumulative speedup over canonical baseline, 14.63s full Middlebury 16-thread benchmark, 100% bit-exact parity).

**P4 shifts focus from micro-benchmarking algorithms to building an embeddable, stream-oriented, resource-controllable, observable, and packageable production stereo engine.**

Key tenets of P4:
1. **Public API Contract**: Separation of product results (`StereoResult`) from internal memory mechanics (`PipelineBuffers`).
2. **Explicit Lifecycle & Workspace**: Workspace allocation isolated from execution pipelines, enabling zero-reallocation steady-state processing without unbounded memory retention.
3. **Bounded Memory Reuse**: Enforcing strict memory budgets (`BoundedReuse`) to prevent long-term memory leaks when processing varying scene resolutions.
4. **Streaming & Interoperability**: Support for continuous frame sessions (`StereoSession`) and raw strided image views (`ImageView8`).
5. **Observability & Diagnostics**: Structured status codes (`Status`), execution diagnostics, and resource accounting.
6. **Packageable Distribution**: Formal CMake install and target export (`apg_sgm::apg_sgm`).
7. **Strict Behavioral Freeze**: 100% bit-exact output parity with P3 (`f906b4d`) across all standard operational modes (`Fast`, `Balanced`, `HighQuality`, `Auto`, `Dense`, `Packed`).

---

## 2. P4 Target Object Architecture

```text
              +--------------------+
              |   PipelineConfig   |
              +--------------------+
                        +
              +--------------------+
              |  ExecutionOptions  |
              +--------------------+
                        |
                        v
              +--------------------+
              |   StereoSession    |
              +--------------------+
                        |
         +--------------+---------------+
         |                              |
         v                              v
+-----------------+            +-----------------+
| StereoInputView |            |  StereoResult   |
+-----------------+            +-----------------+
                                        |
                                        v
                               +-----------------+
                               |  PipelineStats  |
                               +-----------------+
```

### Internal Component Hierarchy

```text
StereoSession
    |
    +-- StereoMatcher (stateless / immutable algorithm runner)
    |
    +-- StereoWorkspace (caller / session managed scratchpad)
            |
            +-- PipelineBuffers (internal intermediate state)
                    |-- Cost Volumes (Dense / Packed)
                    |-- Census Descriptors
                    |-- SGM Accumulators
                    |-- Prior Estimation Maps
```

---

## 3. Scope Boundaries & Explicit Non-Goals

P4 strictly avoids:
- New GPU/CUDA/Vulkan/OpenCL backends.
- Neural stereo or deep learned cost functions.
- Rectification / camera calibration algorithms (inputs must be rectified).
- Delaunay triangulation or planar prior upgrades (deferred to P5).
- Algorithmic Census / SGM / Cost micro-optimizations.
- Breaking source compatibility for the P3 `compute(..., PipelineBuffers&)` interface.
