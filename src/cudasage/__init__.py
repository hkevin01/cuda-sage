"""cuda-sage: CUDA/PTX Static Analysis & Guidance Engine.

Public API — import the components you need:

    from cudasage import PTXParser, OccupancyAnalyzer, DivergenceAnalyzer, MemoryAnalyzer
    from cudasage import get_arch, ARCHITECTURES
    from cudasage.transform import CUDASourceTransformer
    from cudasage.tune import KernelAutoTuner, TuneParam, SearchSpace
"""
# ─────────────────────────────────────────────────────────────────────────────
# Module:         cudasage
# Purpose:        Expose the stable public surface of the cuda-sage library.
# Rationale:      Single import point prevents callers from depending on
#                 internal submodule structure, which may change.
# ─────────────────────────────────────────────────────────────────────────────

__version__ = "0.2.0"

from .parsers.ptx_parser import PTXParser, KernelInfo, RegisterFile, Instruction
from .analyzers.occupancy import OccupancyAnalyzer, OccupancyResult, OccupancyCurvePoint
from .analyzers.divergence import DivergenceAnalyzer, DivergenceResult, DivergenceSite
from .analyzers.memory import MemoryAnalyzer, MemoryResult, SpillWarning, BankConflictRisk
from .models.architectures import Architecture, ARCHITECTURES, get_arch
from .transform import CUDASourceTransformer, TransformResult, Transform
from .tune import TuneParam, SearchSpace, KernelAutoTuner, TuneResult, BenchmarkPoint, TuneCache

__all__ = [
    # Parser
    "PTXParser", "KernelInfo", "RegisterFile", "Instruction",
    # Analyzers
    "OccupancyAnalyzer", "OccupancyResult", "OccupancyCurvePoint",
    "DivergenceAnalyzer", "DivergenceResult", "DivergenceSite",
    "MemoryAnalyzer", "MemoryResult", "SpillWarning", "BankConflictRisk",
    # Architecture models
    "Architecture", "ARCHITECTURES", "get_arch",
    # Source transformer
    "CUDASourceTransformer", "TransformResult", "Transform",
    # Auto-tuner
    "TuneParam", "SearchSpace", "KernelAutoTuner", "TuneResult", "BenchmarkPoint", "TuneCache",
    # Version
    "__version__",
]
