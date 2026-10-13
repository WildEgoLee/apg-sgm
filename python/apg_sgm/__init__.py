"""APG-SGM: High-Performance Dense-to-Sparse Guided Stereo Matching Engine."""

from __future__ import annotations

try:
    from ._core import (
        MemoryPolicy,
        PipelineStats,
        QualityMode,
        StageTiming,
        StatusCode,
        StereoResult,
        StereoSession,
        VolumeBackend,
    )
except ImportError as err:
    raise ImportError(
        "Failed to import apg_sgm C++ extension module '_core'. "
        "Ensure the package is built and installed correctly (e.g. 'pip install .')."
    ) from err

__version__ = "1.0.0"
__all__ = [
    "MemoryPolicy",
    "PipelineStats",
    "QualityMode",
    "StageTiming",
    "StatusCode",
    "StereoResult",
    "StereoSession",
    "VolumeBackend",
    "__version__",
]
