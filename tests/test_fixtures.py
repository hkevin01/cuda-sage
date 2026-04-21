"""Tests for the new PTX fixture kernels: matmul and reduction.

These tests validate that the analyzers correctly handle:
  - Shared memory with sync barriers (matmul, reduction)
  - High register count leading to low occupancy (matmul)
  - Multiple shared memory declarations (matmul: two tiles)
  - Bank conflict detection from literal stride offsets (matmul)
  - Divergence in reduction pattern (reduction)
"""
from __future__ import annotations
import pytest
from cudasage.analyzers.occupancy import OccupancyAnalyzer
from cudasage.analyzers.divergence import DivergenceAnalyzer
from cudasage.analyzers.memory import MemoryAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# matmul fixture
# ─────────────────────────────────────────────────────────────────────────────

def test_matmul_parses_name(matmul_kernel):
    assert matmul_kernel.name == "matmul"


def test_matmul_sm_target(matmul_kernel):
    assert matmul_kernel.sm_target == "sm_80"


def test_matmul_param_count(matmul_kernel):
    assert matmul_kernel.param_count == 4


def test_matmul_has_shared_memory(matmul_kernel):
    assert matmul_kernel.shared_mem_bytes == 2048  # 2 × 1024-byte tiles


def test_matmul_has_sync_barriers(matmul_kernel):
    assert matmul_kernel.sync_barriers >= 2


def test_matmul_no_spills(matmul_kernel):
    assert matmul_kernel.local_loads == 0
    assert matmul_kernel.local_stores == 0


def test_matmul_global_memory_ops(matmul_kernel):
    assert matmul_kernel.global_loads == 2   # A tile + B tile loads
    assert matmul_kernel.global_stores == 1  # C output


def test_matmul_shared_loads_present(matmul_kernel):
    assert matmul_kernel.shared_loads >= 4   # dot-product reads


def test_matmul_occupancy_limited_by_registers(matmul_kernel, arch_sm80):
    result = OccupancyAnalyzer().analyze(matmul_kernel, arch_sm80, 256)
    assert result.limiting_factor == "registers"
    assert result.occupancy > 0.0


def test_matmul_occupancy_range(matmul_kernel, arch_sm80):
    result = OccupancyAnalyzer().analyze(matmul_kernel, arch_sm80, 256)
    assert 0.0 < result.occupancy <= 1.0


def test_matmul_occupancy_curve_has_points(matmul_kernel, arch_sm80):
    curve = OccupancyAnalyzer().occupancy_curve(matmul_kernel, arch_sm80)
    assert len(curve) == 10
    assert all(0.0 <= pt.occupancy <= 1.0 for pt in curve)


def test_matmul_divergence_has_sites(matmul_kernel):
    result = DivergenceAnalyzer().analyze(matmul_kernel)
    # matmul has a bounds check on %tid.x derived coordinates → divergence
    assert result.has_divergence


def test_matmul_memory_no_missing_sync(matmul_kernel):
    result = MemoryAnalyzer().analyze(matmul_kernel)
    # matmul has bar.sync after smem writes → no missing sync warning
    assert not result.possible_missing_sync


def test_matmul_memory_has_bank_conflict_risks(matmul_kernel):
    result = MemoryAnalyzer().analyze(matmul_kernel)
    # Literal small offsets in smem_A/smem_B accesses trigger the heuristic
    assert len(result.bank_conflict_risks) >= 1


def test_matmul_memory_not_spilling(matmul_kernel):
    result = MemoryAnalyzer().analyze(matmul_kernel)
    assert not result.has_spills


def test_matmul_high_register_suggestions(matmul_kernel, arch_sm80):
    result = OccupancyAnalyzer().analyze(matmul_kernel, arch_sm80, 256)
    # High regs → should suggest reducing register usage
    combined = " ".join(result.suggestions).lower()
    assert "register" in combined


# ─────────────────────────────────────────────────────────────────────────────
# reduction fixture
# ─────────────────────────────────────────────────────────────────────────────

def test_reduction_parses_name(reduction_kernel):
    assert reduction_kernel.name == "reduce_sum"


def test_reduction_param_count(reduction_kernel):
    assert reduction_kernel.param_count == 3


def test_reduction_has_shared_memory(reduction_kernel):
    assert reduction_kernel.shared_mem_bytes == 1024


def test_reduction_has_sync_barriers(reduction_kernel):
    assert reduction_kernel.sync_barriers >= 2


def test_reduction_no_spills(reduction_kernel):
    assert reduction_kernel.local_loads == 0
    assert reduction_kernel.local_stores == 0


def test_reduction_global_ops(reduction_kernel):
    assert reduction_kernel.global_loads >= 1
    assert reduction_kernel.global_stores == 1


def test_reduction_occupancy_positive(reduction_kernel, arch_sm80):
    result = OccupancyAnalyzer().analyze(reduction_kernel, arch_sm80, 256)
    assert result.occupancy > 0.0


def test_reduction_divergence_detected(reduction_kernel):
    result = DivergenceAnalyzer().analyze(reduction_kernel)
    # Reduction steps branch on %tid.x comparisons → divergent branches
    assert result.has_divergence


def test_reduction_memory_no_missing_sync(reduction_kernel):
    result = MemoryAnalyzer().analyze(reduction_kernel)
    # reduction has bar.sync before reads → no missing sync warning
    assert not result.possible_missing_sync


def test_reduction_shared_loads_present(reduction_kernel):
    assert reduction_kernel.shared_loads >= 2


# ─────────────────────────────────────────────────────────────────────────────
# diff: vecadd vs matmul (different kernels → no common names)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_common_kernels_between_vecadd_and_matmul(vecadd_kernel, matmul_kernel):
    """vecadd and matmul have different names; diff should find no common kernels."""
    vecadd_names = {vecadd_kernel.name}
    matmul_names = {matmul_kernel.name}
    common = vecadd_names & matmul_names
    assert len(common) == 0
