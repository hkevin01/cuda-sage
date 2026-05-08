"""Memory access pattern analyzer.

Detects:
  1. Register spills to local memory (ld.local / st.local) — worst-case
     memory access, functionally equivalent to global memory latency.
  2. Shared memory bank conflict risk from access stride patterns.
  3. Instruction mix: compute-to-memory ratio for bounding analysis.
  4. Synchronization sufficiency (shared mem writes not followed by bar.sync).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from ..parsers.ptx_parser import KernelInfo


@dataclass
class SpillWarning:
    kind: str        # "load" | "store"
    count: int
    severity: str    # "critical" | "warning"


@dataclass
class BankConflictRisk:
    """Heuristic: shared memory accessed with large constant stride risks conflicts."""
    access_pattern: str
    stride_hint: int | None    # None = unknown
    risk_level: str            # "high" | "medium" | "low"
    description: str


@dataclass
class MemoryResult:
    kernel_name: str
    spill_warnings: list[SpillWarning] = field(default_factory=list)
    bank_conflict_risks: list[BankConflictRisk] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Instruction mix metrics
    global_load_count: int = 0
    global_store_count: int = 0
    shared_load_count: int = 0
    shared_store_count: int = 0
    arithmetic_count: int = 0

    # Synchronization
    shared_writes_before_sync: int = 0
    sync_barriers: int = 0
    possible_missing_sync: bool = False

    @property
    def has_spills(self) -> bool:
        return bool(self.spill_warnings)

    @property
    def memory_bound_likely(self) -> bool:
        """True when global memory ops dominate over arithmetic."""
        total_ops = self.arithmetic_count + self.global_load_count + self.global_store_count
        if total_ops == 0:
            return False
        mem_ratio = (self.global_load_count + self.global_store_count) / total_ops
        return mem_ratio > 0.4

    @property
    def arithmetic_intensity_proxy(self) -> float:
        mem_ops = self.global_load_count + self.global_store_count
        if mem_ops == 0:
            return float("inf")
        return self.arithmetic_count / mem_ops


# Detect stride patterns in shared memory indexing (e.g., smem[tid * N])
_RE_SHARED_STRIDE = re.compile(r"(?:ld|st)\.shared\S*\s+.*\[.*\+\s*(\d+)\]")
_BANK_WIDTH = 32  # 32-bit bank width; 32 banks → conflicts at stride == 32


class MemoryAnalyzer:
    """Analyze memory access patterns from a parsed PTX kernel.

    Checks performed:
      1. Register spills (ld.local / st.local) — ~600 cycle latency each.
      2. Shared memory bank conflict risk (stride-based heuristic).
      3. Arithmetic intensity proxy (ops / global-mem-op).
      4. Missing __syncthreads() after shared memory writes.

    Design:         Pure computation over KernelInfo counts; no I/O.
    Thread Safety:  Stateless; reentrant.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         analyze
    # Purpose:        Build a MemoryResult from KernelInfo instruction counts.
    # Inputs:         kernel: KernelInfo — fully parsed PTX kernel
    # Outputs:        MemoryResult with spill_warnings, bank_conflict_risks,
    #                 sync warning flag, intensity proxy, and suggestions.
    # Side Effects:   None.
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, kernel: KernelInfo) -> MemoryResult:
        result = MemoryResult(kernel_name=kernel.name)

        # ── Copy instruction counts from KernelInfo ───────────────────────
        result.global_load_count  = kernel.global_loads
        result.global_store_count = kernel.global_stores
        result.shared_load_count  = kernel.shared_loads
        result.shared_store_count = kernel.shared_stores
        result.arithmetic_count   = kernel.arithmetic
        result.sync_barriers      = kernel.sync_barriers

        # ── Spill detection ───────────────────────────────────────────────
        if kernel.local_loads > 0:
            result.spill_warnings.append(SpillWarning(
                kind="load", count=kernel.local_loads,
                severity="critical" if kernel.local_loads > 10 else "warning",
            ))
        if kernel.local_stores > 0:
            result.spill_warnings.append(SpillWarning(
                kind="store", count=kernel.local_stores,
                severity="critical" if kernel.local_stores > 10 else "warning",
            ))

        # ── Shared memory bank conflict heuristic ─────────────────────────
        result.bank_conflict_risks = self._detect_bank_conflicts(kernel)

        # ── Sync barrier sufficiency ──────────────────────────────────────
        # If shared memory writes exist but no bar.sync → likely missing sync
        total_smem_writes = kernel.shared_stores
        if total_smem_writes > 0 and kernel.sync_barriers == 0:
            result.possible_missing_sync = True
            result.shared_writes_before_sync = total_smem_writes

        result.suggestions = self._suggest(result, kernel)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         _detect_bank_conflicts
    # Purpose:        Scan shared memory instructions for stride-based conflict risk.
    # Inputs:         kernel: KernelInfo
    # Outputs:        list[BankConflictRisk] — one entry per risky instruction
    # Rationale:      32 banks × 4-byte width means stride-32 accesses cause
    #                 32-way serialization. Power-of-two strides divisible by 8
    #                 cause proportionally smaller but still significant conflicts.
    # Failure Modes:  Heuristic may produce false positives; full analysis
    #                 requires runtime profiling with Nsight Compute.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_bank_conflicts(kernel: KernelInfo) -> list[BankConflictRisk]:
        risks: list[BankConflictRisk] = []
        if not kernel.instructions:
            return risks

        for instr in kernel.instructions:
            full = f"{instr.opcode} {instr.operands}"
            if not ("ld.shared" in full or "st.shared" in full):
                continue
            m = _RE_SHARED_STRIDE.search(full)
            if m:
                try:
                    stride = int(m.group(1))
                except ValueError:
                    continue
                # Stride of 0 or 1 → no conflict; stride == 32 → 32-way conflict
                if stride == 0 or stride == 1:
                    continue
                # Power-of-two strides that are multiples of bank count are worst
                conflict_level = "low"
                if stride % _BANK_WIDTH == 0:
                    conflict_level = "high"
                elif stride % 8 == 0:
                    conflict_level = "medium"
                risks.append(BankConflictRisk(
                    access_pattern=full.strip(),
                    stride_hint=stride,
                    risk_level=conflict_level,
                    description=(
                        f"Shared memory accessed with stride={stride} bytes. "
                        f"{'32-way bank conflict — worst case!' if conflict_level == 'high' else 'Possible bank conflicts.'}"
                    ),
                ))
        return risks

    @staticmethod
    def _suggest(result: MemoryResult, kernel: KernelInfo) -> list[str]:
        suggestions = []

        # Spills
        if result.has_spills:
            total_spills = sum(w.count for w in result.spill_warnings)
            suggestions.append(
                f"Register spilling detected: {total_spills} local memory operation(s). "
                "Local memory has global memory latency (~600 cycles). "
                "Reduce register pressure: use __launch_bounds__, split the kernel, "
                "or store intermediate values in shared memory."
            )

        # Bank conflicts
        high_bc = [r for r in result.bank_conflict_risks if r.risk_level == "high"]
        if high_bc:
            suggestions.append(
                f"{len(high_bc)} potential 32-way shared memory bank conflict(s). "
                "Pad shared memory arrays by 1 element to break the stride: "
                "__shared__ float smem[ROWS][COLS + 1];"
            )

        # Missing sync
        if result.possible_missing_sync:
            suggestions.append(
                f"{result.shared_writes_before_sync} shared memory write(s) with no "
                "__syncthreads() / bar.sync detected. Threads may read stale data. "
                "Add __syncthreads() after writing and before reading shared memory."
            )

        # Compute vs memory bound
        if result.memory_bound_likely:
            ai = result.arithmetic_intensity_proxy
            suggestions.append(
                f"Low arithmetic intensity ({ai:.1f} ops/mem op). Kernel is likely "
                "memory-bandwidth bound. Consider: increasing data reuse via tiling, "
                "fusing adjacent kernels, or using vectorized loads (ld.global.v4.f32)."
            )
        elif result.arithmetic_intensity_proxy > 10:
            suggestions.append(
                f"High arithmetic intensity ({result.arithmetic_intensity_proxy:.1f} ops/mem op). "
                "Kernel is likely compute-bound — focus on occupancy and IPC rather than memory."
            )

        # Special function suggestion
        if kernel.special_fns > 0:
            suggestions.append(
                f"{kernel.special_fns} special function unit (SFU) instruction(s) (sin/cos/sqrt). "
                "SFUs have limited throughput (8/SM on most architectures). "
                "Consider __fast_math__ approximations or table lookups for hot paths."
            )

        return suggestions
