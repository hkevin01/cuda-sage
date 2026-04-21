"""CUDA kernel auto-tuner: NVRTC compilation + CUDA-events benchmarking.

Two execution paths:
  GPU path  (requires cuda-python ≥ 12):
    1. Substitute #define values into template source
    2. Compile to PTX with NVRTC
    3. Load PTX via cuModuleLoadData
    4. Allocate device buffers, set up kernel args
    5. Warm-up N times, then measure M runs with cuEventElapsedTime
    6. Record median latency as the score for this config

  Static model path  (no GPU / no cuda-python):
    1. Substitute #define values into template source
    2. Extract inferred register count heuristic from source complexity
    3. Compute occupancy score via the existing OccupancyAnalyzer formula
    4. Use inverse occupancy-latency model: score ∝ 1/occupancy

The GPU path is authoritative; the static path is useful for pre-screening
large search spaces before committing to GPU time.

cuda-python: https://github.com/NVIDIA/cuda-python
NVRTC guide:  https://docs.nvidia.com/cuda/nvrtc/
"""
from __future__ import annotations
import hashlib
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ..models.architectures import get_arch
from ..analyzers.occupancy import OccupancyAnalyzer

# Optional NVIDIA cuda-python import
try:
    from cuda import cuda as _cuda_driver
    from cuda import nvrtc as _nvrtc
    _HAS_CUDA = True
except ImportError:
    _cuda_driver = None  # type: ignore[assignment]
    _nvrtc = None        # type: ignore[assignment]
    _HAS_CUDA = False


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkPoint:
    """Result for a single configuration in the search space."""
    params: dict[str, Any]
    time_ms: float                       # median kernel time (or model score in ms-equivalent)
    occupancy: float                     # predicted occupancy 0-1
    source: Literal["gpu", "model"]      # "gpu" = real timing; "model" = static estimate
    error: Optional[str] = None          # compilation / runtime error if any


@dataclass
class TuneResult:
    """Output of KernelAutoTuner.tune().

    best_params: Configuration with lowest time_ms (or highest model score).
    speedup:     time(default) / time(best).  > 1.0 means improvement.
    all_points:  Every evaluated configuration (for analysis / plotting).
    recommendations: Human-readable improvement notes.
    """
    kernel_name: str
    arch: str
    best_params: dict[str, Any]
    best_time_ms: float
    default_time_ms: float
    speedup: float                        # > 1.0 → we found something better
    all_points: list[BenchmarkPoint] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    source: Literal["gpu", "model"] = "model"

    @property
    def improved(self) -> bool:
        return self.speedup > 1.05     # >5% improvement threshold


# ─────────────────────────────────────────────────────────────────────────────
# NVRTC helpers (GPU path)
# ─────────────────────────────────────────────────────────────────────────────

def _check(err: Any) -> None:
    """Raise on CUDA/NVRTC error codes."""
    if isinstance(err, tuple):
        err = err[0]
    # cuda-python returns (error_code, result) tuples; error code 0 = success
    if hasattr(err, 'value') and err.value != 0:
        raise RuntimeError(f"CUDA error: {err}")


def _nvrtc_compile(source: str, name: str, arch: str) -> bytes:
    """Compile CUDA C source → PTX bytes using NVRTC.

    Inputs:
        source: CUDA C/C++ kernel source as a string.
        name:   Logical filename for error messages (e.g. 'kernel.cu').
        arch:   SM target string, e.g. 'sm_86'.

    Returns: PTX assembly as bytes.
    Raises:  RuntimeError with NVRTC log on compilation failure.
    """
    err, prog = _nvrtc.nvrtcCreateProgram(
        source.encode(), name.encode(), 0, [], []
    )
    _check(err)

    opts = [
        f"--gpu-architecture={arch}".encode(),
        b"--std=c++17",
        b"--generate-line-info",
        b"-O3",
    ]
    err, = _nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err.value != 0:
        _, log_size = _nvrtc.nvrtcGetProgramLogSize(prog)
        _, log = _nvrtc.nvrtcGetProgramLog(prog, log_size)
        raise RuntimeError(f"NVRTC compilation failed:\n{log.decode()}")

    _, ptx_size = _nvrtc.nvrtcGetPTXSize(prog)
    _, ptx = _nvrtc.nvrtcGetPTX(prog, ptx_size)
    _nvrtc.nvrtcDestroyProgram(prog)
    return ptx


def _gpu_benchmark(
    ptx: bytes,
    kernel_name: str,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args_flat: list[Any],
    warmup: int = 3,
    runs: int = 10,
) -> float:
    """Load PTX, launch kernel, return median time in ms via CUDA events.

    Inputs:
        ptx:         PTX bytes from NVRTC.
        kernel_name: Entry point name (must match .entry in PTX).
        grid:        Grid dimensions (gx, gy, gz).
        block:       Block dimensions (bx, by, bz).
        args_flat:   Kernel arguments as a list of ctypes values or device pointers.
        warmup:      Number of un-timed warm-up launches.
        runs:        Number of timed launches; median is returned.

    Returns: Median kernel execution time in milliseconds.
    """
    import ctypes
    _, ctx = _cuda_driver.cuCtxGetCurrent()

    err, module = _cuda_driver.cuModuleLoadData(ptx)
    _check(err)
    err, func = _cuda_driver.cuModuleGetFunction(module, kernel_name.encode())
    _check(err)

    err, stream = _cuda_driver.cuStreamCreate(0)
    _check(err)
    err, t_start = _cuda_driver.cuEventCreate(0)
    _check(err)
    err, t_stop = _cuda_driver.cuEventCreate(0)
    _check(err)

    # Build kernel arg pointers
    arg_ptrs = [ctypes.c_void_p(ctypes.addressof(a)) for a in args_flat]
    kernel_args = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)

    # Warm-up
    for _ in range(warmup):
        _check(_cuda_driver.cuLaunchKernel(
            func, *grid, *block, 0, stream, kernel_args, 0
        ))
    _check(_cuda_driver.cuStreamSynchronize(stream))

    # Timed runs
    times: list[float] = []
    for _ in range(runs):
        _check(_cuda_driver.cuEventRecord(t_start, stream))
        _check(_cuda_driver.cuLaunchKernel(
            func, *grid, *block, 0, stream, kernel_args, 0
        ))
        _check(_cuda_driver.cuEventRecord(t_stop, stream))
        _check(_cuda_driver.cuStreamSynchronize(stream))
        err, ms = _cuda_driver.cuEventElapsedTime(t_start, t_stop)
        _check(err)
        times.append(ms)

    _cuda_driver.cuModuleUnload(module)
    _cuda_driver.cuStreamDestroy(stream)
    _cuda_driver.cuEventDestroy(t_start)
    _cuda_driver.cuEventDestroy(t_stop)

    return statistics.median(times)


# ─────────────────────────────────────────────────────────────────────────────
# Static model (no-GPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_register_count(source: str) -> int:
    """Heuristic register count from CUDA C source complexity.

    This is intentionally coarse — it only needs to distinguish
    high/medium/low register pressure configurations for relative scoring.

    Inputs:
        source: CUDA C source with #defines already substituted.
    Returns:
        Estimated registers per thread (int, 8–128).
    """
    # Count unique variable names in kernel body
    body_match = re.search(r"\{(.*)\}", source, re.DOTALL)
    if not body_match:
        return 32
    body = body_match.group(1)

    # Simple heuristics
    float_vars = len(re.findall(r"\bfloat\b", body))
    double_vars = len(re.findall(r"\bdouble\b", body))
    int_vars = len(re.findall(r"\bint\b", body))
    loops = len(re.findall(r"\bfor\b", body))

    estimate = 8 + float_vars + double_vars * 2 + int_vars // 2 + loops * 2
    return min(max(estimate, 8), 128)


def _static_score(source: str, block_size: int, arch_str: str) -> tuple[float, float]:
    """Return (latency_ms_equiv, occupancy) using the occupancy cost model.

    The latency model is:
        effective_cycles = base_cycles / occupancy
    where base_cycles is a fixed normalisation constant (1000 ms-equiv).
    This captures the key insight that higher occupancy hides memory latency
    through warp switching.

    Inputs:
        source:     CUDA C source with #defines substituted.
        block_size: Threads per block for this configuration.
        arch_str:   SM target string (e.g. 'sm_80').
    Returns:
        (time_ms_equiv, occupancy)  — time is relative, not absolute.
    """
    arch = get_arch(arch_str)
    regs = _infer_register_count(source)

    # Shared mem estimate from __shared__ declarations
    smem = 0
    for m in re.finditer(r"__shared__\s+\w+\s+\w+\[(\d+)\]", source):
        # Assume 4 bytes/element if float; overestimate is safe
        smem += int(m.group(1)) * 4

    result = OccupancyAnalyzer().analyze_raw(
        threads_per_block=block_size,
        regs_per_thread=regs,
        shared_mem_bytes=smem,
        arch=arch,
    )
    occ = result.occupancy if result.occupancy > 0 else 0.01
    time_equiv = 1000.0 / occ   # lower is better
    return time_equiv, occ


# ─────────────────────────────────────────────────────────────────────────────
# Auto-tuner
# ─────────────────────────────────────────────────────────────────────────────

class KernelAutoTuner:
    """Finds the optimal launch configuration for a CUDA kernel template.

    Inputs (to .tune()):
        source:      CUDA C source with tunable #define parameters.
        kernel_name: Entry point name (used for NVRTC + cache key).
        space:       SearchSpace defining parameters and their candidate values.
        arch:        SM target string (e.g. 'sm_86').
        n:           Problem size (used for grid computation in GPU path).
        cache:       Optional TuneCache for persistent result storage.
        force_model: If True, skip GPU even if available (testing).

    Outputs:
        TuneResult with best_params, speedup, all_points, recommendations.

    Side Effects:
        If cache is provided, results are stored after completion.
        GPU memory is allocated and freed within each benchmark call.
    """

    def tune(
        self,
        source: str,
        kernel_name: str,
        space: "SearchSpace",
        arch: str = "sm_80",
        n: int = 1_000_000,
        cache: "Optional[TuneCache]" = None,
        force_model: bool = False,
    ) -> TuneResult:
        from .cache import TuneCache

        use_gpu = _HAS_CUDA and not force_model
        configs = space.configs()
        default_cfg = space.default_config()

        points: list[BenchmarkPoint] = []
        default_time: Optional[float] = None

        # Initialise CUDA context once for the GPU path
        ctx = None
        if use_gpu:
            try:
                _check(_cuda_driver.cuInit(0))
                _, dev = _cuda_driver.cuDeviceGet(0)
                _, ctx = _cuda_driver.cuCtxCreate(0, dev)
            except Exception as exc:
                use_gpu = False

        for cfg in configs:
            # Check cache
            if cache:
                cached = cache.get(source, arch, cfg)
                if cached is not None:
                    points.append(cached)
                    if cfg == default_cfg:
                        default_time = cached.time_ms
                    continue

            rendered = _substitute(source, cfg)

            if use_gpu:
                pt = self._benchmark_gpu(rendered, kernel_name, arch, cfg, n)
            else:
                pt = self._benchmark_model(rendered, cfg, arch)

            points.append(pt)
            if cache:
                cache.put(source, arch, cfg, pt)
            if cfg == default_cfg:
                default_time = pt.time_ms

        if ctx is not None:
            try:
                _cuda_driver.cuCtxDestroy(ctx)
            except Exception:
                pass

        # Find best (minimum time)
        valid = [p for p in points if p.error is None]
        if not valid:
            raise RuntimeError("All configurations failed — check source for errors")

        best = min(valid, key=lambda p: p.time_ms)
        default_time = default_time or valid[0].time_ms
        speedup = default_time / best.time_ms if best.time_ms > 0 else 1.0

        result = TuneResult(
            kernel_name=kernel_name,
            arch=arch,
            best_params=best.params,
            best_time_ms=best.time_ms,
            default_time_ms=default_time,
            speedup=speedup,
            all_points=points,
            source="gpu" if use_gpu else "model",
        )
        result.recommendations = self._recommend(result)
        return result

    def _benchmark_gpu(
        self,
        source: str,
        kernel_name: str,
        arch: str,
        params: dict[str, Any],
        n: int,
    ) -> BenchmarkPoint:
        import ctypes, math
        block_size = params.get("BLOCK_SIZE", 256)
        grid_size = math.ceil(n / block_size)

        try:
            ptx = _nvrtc_compile(source, f"{kernel_name}.cu", arch)
        except RuntimeError as exc:
            return BenchmarkPoint(
                params=params, time_ms=float("inf"),
                occupancy=0.0, source="gpu", error=str(exc)
            )

        try:
            # Allocate minimal dummy buffers (GPU path needs real allocations
            # for timing, but zero-fill is fine for benchmarking purposes)
            buf_size = n * ctypes.sizeof(ctypes.c_float)
            _, d_a = _cuda_driver.cuMemAlloc(buf_size)
            _, d_b = _cuda_driver.cuMemAlloc(buf_size)
            _, d_c = _cuda_driver.cuMemAlloc(buf_size)
            c_n = ctypes.c_int(n)

            # Build args: (float* a, float* b, float* c, int n)
            args = [
                ctypes.c_size_t(int(d_a)),
                ctypes.c_size_t(int(d_b)),
                ctypes.c_size_t(int(d_c)),
                c_n,
            ]
            time_ms = _gpu_benchmark(
                ptx, kernel_name,
                grid=(grid_size, 1, 1), block=(block_size, 1, 1),
                args_flat=args,
            )
            for ptr in (d_a, d_b, d_c):
                _cuda_driver.cuMemFree(ptr)

            # Occupancy from PTX
            from ..parsers.ptx_parser import PTXParser
            kernels = PTXParser().parse_string(ptx.decode(errors="replace"))
            occ = 0.0
            if kernels:
                from ..analyzers.occupancy import OccupancyAnalyzer
                arch_spec = get_arch(arch)
                occ = OccupancyAnalyzer().analyze(kernels[0], arch_spec, block_size).occupancy

            return BenchmarkPoint(params=params, time_ms=time_ms, occupancy=occ, source="gpu")
        except Exception as exc:
            return BenchmarkPoint(
                params=params, time_ms=float("inf"),
                occupancy=0.0, source="gpu", error=str(exc)
            )

    def _benchmark_model(
        self,
        source: str,
        params: dict[str, Any],
        arch: str,
    ) -> BenchmarkPoint:
        block_size = params.get("BLOCK_SIZE", 256)
        time_equiv, occ = _static_score(source, block_size, arch)
        return BenchmarkPoint(
            params=params,
            time_ms=time_equiv,
            occupancy=occ,
            source="model",
        )

    @staticmethod
    def _recommend(result: TuneResult) -> list[str]:
        recs: list[str] = []
        if result.improved:
            diff_pct = (result.speedup - 1.0) * 100
            recs.append(
                f"Best config achieves {diff_pct:.1f}% predicted improvement over default. "
                f"Apply: {', '.join(f'{k}={v}' for k,v in result.best_params.items())}"
            )
        else:
            recs.append(
                "Default configuration is near-optimal for this architecture. "
                "Consider kernel fusion or memory access pattern changes for further gains."
            )

        # Flag any configs that failed to compile
        errors = [p for p in result.all_points if p.error]
        if errors:
            recs.append(
                f"{len(errors)} configuration(s) failed to compile. "
                "Check source for block-size-dependent shared memory bounds."
            )

        # Suggest grid search if random was used and space is large
        if len(result.all_points) < 6:
            recs.append(
                "Small search space — consider adding more parameter values "
                "(e.g. TILE_SIZE) for a wider optimisation sweep."
            )

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _substitute(source: str, params: dict[str, Any]) -> str:
    """Replace #define NAME VALUE with #define NAME <new_value> for each param."""
    for name, value in params.items():
        source = re.sub(
            rf"(#define\s+{re.escape(name)}\s+)\S+",
            rf"\g<1>{value}",
            source,
        )
    return source
