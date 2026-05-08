"""Occupancy calculator for CUDA kernels.

Implements the standard CUDA occupancy formula:
  achieved_occupancy = active_warps_per_sm / max_warps_per_sm

Limiting factors (whichever allows fewest concurrent blocks wins):
  - Thread count
  - Register file capacity
  - Shared memory capacity
  - Maximum blocks per SM hardware limit
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from ..models.architectures import Architecture
from ..parsers.ptx_parser import KernelInfo


@dataclass
class OccupancyResult:
    threads_per_block: int
    regs_per_thread: int
    shared_mem_bytes: int
    arch: str

    active_blocks: int = 0
    active_warps: int = 0
    occupancy: float = 0.0          # 0.0 – 1.0
    limiting_factor: str = "unknown"

    # Per-factor block limits for diagnostics
    blocks_by_threads: int = 0
    blocks_by_regs: int = 0
    blocks_by_smem: int = 0
    blocks_by_hw_limit: int = 0

    suggestions: list[str] = field(default_factory=list)


@dataclass
class OccupancyCurvePoint:
    threads_per_block: int
    occupancy: float
    limiting_factor: str


class OccupancyAnalyzer:
    """Calculate theoretical occupancy for CUDA kernels.

    Implements the standard CUDA occupancy formula:
        achieved_occupancy = active_warps_per_sm / max_warps_per_sm

    where active_warps_per_sm = active_blocks × warps_per_block, and
    active_blocks = min(thread_limit, reg_limit, smem_limit, hw_block_limit).

    Design:         Pure computation; no state, no side effects.
    Thread Safety:  All methods are stateless and reentrant.
    References:     CUDA C Programming Guide Appendix G (Compute Capabilities),
                    CUDA Occupancy Calculator spreadsheet (NVIDIA, 2023).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         analyze
    # Purpose:        Compute occupancy for a single kernel/arch/block-size.
    # Inputs:         kernel: KernelInfo — parsed PTX metadata
    #                 arch: Architecture — SM hardware constraints
    #                 threads_per_block: int — launch configuration [1..1024]
    # Outputs:        OccupancyResult with occupancy in [0.0, 1.0] and the
    #                 identifying limiting_factor.
    # Postconditions: result.occupancy == result.active_warps / arch.max_warps_per_sm
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(
        self,
        kernel: KernelInfo,
        arch: Architecture,
        threads_per_block: int = 256,
    ) -> OccupancyResult:
        regs = kernel.registers.physical_regs
        smem = kernel.shared_mem_bytes
        return self._calculate(arch, threads_per_block, regs, smem)

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         occupancy_curve
    # Purpose:        Sweep block sizes and return occupancy at each point.
    # Inputs:         kernel: KernelInfo
    #                 arch: Architecture
    #                 thread_counts: list[int] | None — sizes to sweep;
    #                     defaults to [32,64,96,128,192,256,384,512,768,1024]
    # Outputs:        list[OccupancyCurvePoint] in the same order as thread_counts
    # ─────────────────────────────────────────────────────────────────────────
    def occupancy_curve(
        self,
        kernel: KernelInfo,
        arch: Architecture,
        thread_counts: list[int] | None = None,
    ) -> list[OccupancyCurvePoint]:
        """Return occupancy for a sweep of thread-block sizes."""
        if thread_counts is None:
            thread_counts = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        # Guard invalid/duplicate thread counts to keep reports deterministic.
        thread_counts = [t for t in dict.fromkeys(thread_counts) if isinstance(t, int) and t > 0]
        if not thread_counts:
            return []

        regs = kernel.registers.physical_regs
        smem = kernel.shared_mem_bytes
        curve = []
        for tpb in thread_counts:
            result = self._calculate(arch, tpb, regs, smem)
            curve.append(OccupancyCurvePoint(tpb, result.occupancy, result.limiting_factor))
        return curve

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         _calculate
    # Purpose:        Core occupancy formula: compute the four resource limits
    #                 and return the minimum (the actual bottleneck).
    # Inputs:         arch: Architecture — hardware limits
    #                 threads_per_block: int — launch config
    #                 regs_per_thread: int ≥ 0 — from PTX .reg declarations
    #                 shared_mem_bytes: int ≥ 0 — per-block shared memory
    # Outputs:        OccupancyResult — all four block-count limits populated
    # Assumptions:    arch.warp_size == 32 for all supported architectures.
    # Constraints:    threads_per_block == 0 → occupancy = 0.0 (safe return).
    # ─────────────────────────────────────────────────────────────────────────
    def _calculate(
        self,
        arch: Architecture,
        threads_per_block: int,
        regs_per_thread: int,
        shared_mem_bytes: int,
    ) -> OccupancyResult:
        if arch.max_warps_per_sm <= 0 or arch.max_threads_per_sm <= 0:
            return OccupancyResult(
                threads_per_block=threads_per_block,
                regs_per_thread=regs_per_thread,
                shared_mem_bytes=shared_mem_bytes,
                arch=arch.sm,
                limiting_factor="arch_invalid",
            )

        ws = arch.warp_size
        result = OccupancyResult(
            threads_per_block=threads_per_block,
            regs_per_thread=regs_per_thread,
            shared_mem_bytes=shared_mem_bytes,
            arch=arch.sm,
            blocks_by_hw_limit=arch.max_blocks_per_sm,
        )

        if threads_per_block <= 0 or threads_per_block > arch.max_threads_per_sm:
            result.limiting_factor = "threads_per_block_invalid"
            return result

        warps_per_block = math.ceil(threads_per_block / ws)

        # ── Limit 1: Thread count ─────────────────────────────────────────
        result.blocks_by_threads = arch.max_threads_per_sm // threads_per_block

        # ── Limit 2: Register file ────────────────────────────────────────
        if regs_per_thread > 0:
            # Registers are allocated per-warp in multiples of reg_alloc_granularity
            # effective_regs_per_thread is rounded up to alloc granularity
            gran = arch.reg_alloc_granularity
            eff_regs = math.ceil(regs_per_thread / gran) * gran
            regs_per_block = eff_regs * warps_per_block * ws
            result.blocks_by_regs = max(0, arch.regs_per_sm // regs_per_block) if regs_per_block else arch.max_blocks_per_sm
        else:
            result.blocks_by_regs = arch.max_blocks_per_sm

        # ── Limit 3: Shared memory ────────────────────────────────────────
        if shared_mem_bytes > 0:
            smem_gran = arch.smem_alloc_granularity
            eff_smem = math.ceil(shared_mem_bytes / smem_gran) * smem_gran
            result.blocks_by_smem = arch.smem_per_sm_bytes // eff_smem
        else:
            result.blocks_by_smem = arch.max_blocks_per_sm

        # ── Final active blocks = minimum of all limits ───────────────────
        active_blocks = min(
            result.blocks_by_threads,
            result.blocks_by_regs,
            result.blocks_by_smem,
            result.blocks_by_hw_limit,
        )
        result.active_blocks = max(0, active_blocks)
        result.active_warps = result.active_blocks * warps_per_block
        result.occupancy = result.active_warps / arch.max_warps_per_sm

        # Identify the bottleneck
        min_val = result.active_blocks
        if min_val == result.blocks_by_regs and regs_per_thread > 0:
            result.limiting_factor = "registers"
        elif min_val == result.blocks_by_smem and shared_mem_bytes > 0:
            result.limiting_factor = "shared_memory"
        elif min_val == result.blocks_by_threads:
            result.limiting_factor = "threads_per_block"
        else:
            result.limiting_factor = "hw_block_limit"

        # ── Suggestions ───────────────────────────────────────────────────
        result.suggestions = self._suggest(result, arch, regs_per_thread, shared_mem_bytes)
        return result

    @staticmethod
    def _suggest(
        r: OccupancyResult,
        arch: Architecture,
        regs: int,
        smem: int,
    ) -> list[str]:
        suggestions = []

        if r.occupancy < 0.25:
            suggestions.append(
                f"Very low occupancy ({r.occupancy:.0%}). The GPU will have trouble hiding "
                "memory latency. Consider restructuring the kernel."
            )
        elif r.occupancy < 0.5:
            suggestions.append(
                f"Below-median occupancy ({r.occupancy:.0%}). Latency hiding may be insufficient."
            )

        if r.limiting_factor == "registers":
            suggestions.append(
                f"Register pressure is the bottleneck ({regs} regs/thread). "
                "Try: __launch_bounds__(threads, minblocks) to cap register usage, "
                "or split the kernel into smaller phases."
            )
            if regs > 64:
                suggestions.append(
                    f"{regs} registers/thread is high. Reducing to ≤32 would double active warps. "
                    "Profile for unnecessary temporaries or consider -maxrregcount."
                )

        if r.limiting_factor == "shared_memory":
            suggestions.append(
                f"Shared memory ({smem / 1024:.1f} KB/block) is the bottleneck. "
                "Consider using dynamic shared memory with cudaFuncSetAttribute to adjust "
                "smem/L1 split, or tile smaller to reduce smem footprint."
            )

        if r.limiting_factor == "threads_per_block":
            # Suggest a block size that might improve occupancy
            best_tpb = r.threads_per_block
            best_occ = r.occupancy
            for tpb in [64, 128, 256, 512]:
                warps = math.ceil(tpb / arch.warp_size)
                active = min(arch.max_threads_per_sm // tpb, arch.max_blocks_per_sm)
                occ = (active * warps) / arch.max_warps_per_sm
                if occ > best_occ:
                    best_occ, best_tpb = occ, tpb
            if best_tpb != r.threads_per_block:
                suggestions.append(
                    f"Changing block size from {r.threads_per_block} → {best_tpb} threads "
                    f"may raise occupancy from {r.occupancy:.0%} → {best_occ:.0%}."
                )

        return suggestions
