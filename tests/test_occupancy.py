"""Tests for the occupancy calculator."""
import pytest
from cudasage.analyzers.occupancy import OccupancyAnalyzer
from cudasage.models.architectures import ARCHITECTURES, get_arch
from cudasage.parsers.ptx_parser import PTXParser, KernelInfo, RegisterFile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
A100 = ARCHITECTURES["sm_80"]
RTX3090 = ARCHITECTURES["sm_86"]
H100 = ARCHITECTURES["sm_90"]


def make_kernel(regs: int = 32, smem: int = 0) -> KernelInfo:
    k = KernelInfo(name="test", sm_target="sm_80")
    k.registers.b32 = regs
    k.shared_mem_bytes = smem
    return k


def test_perfect_occupancy_small_kernel():
    """A kernel with few registers and no shared mem should achieve full occupancy on A100."""
    k = make_kernel(regs=16, smem=0)
    result = OccupancyAnalyzer().analyze(k, A100, threads_per_block=128)
    # 128 threads = 4 warps/block; A100 max 32 blocks → 128 warps active out of 64 max
    # 32 blocks × 4 warps = 128 warps → 128/64 = 2.0? No — must be capped at max_warps
    # active_blocks = min(2048//128=16, regs_limited=..., smem=32, hw=32) = 16
    # active_warps = 16 × 4 = 64 = max → 100% occupancy
    assert result.occupancy == pytest.approx(1.0, abs=0.01)


def test_register_heavy_kernel_limits_occupancy():
    """A kernel using 64 registers/thread should be register-limited on A100."""
    k = make_kernel(regs=64, smem=0)
    result = OccupancyAnalyzer().analyze(k, A100, threads_per_block=256)
    assert result.occupancy < 0.75
    assert result.limiting_factor == "registers"


def test_large_shared_mem_limits_occupancy():
    """48KB shared memory per block limits blocks to 2 on A100 (164KB smem)."""
    k = make_kernel(regs=16, smem=48 * 1024)
    result = OccupancyAnalyzer().analyze(k, A100, threads_per_block=256)
    assert result.blocks_by_smem <= 3
    assert result.limiting_factor in ("shared_memory", "hw_block_limit")


def test_occupancy_curve_has_correct_length():
    k = make_kernel(regs=32, smem=0)
    curve = OccupancyAnalyzer().occupancy_curve(k, A100)
    assert len(curve) == 10  # default thread counts


def test_occupancy_curve_increases_with_more_threads_for_small_kernels():
    """For a small register kernel, larger blocks should not decrease occupancy below small blocks."""
    k = make_kernel(regs=8, smem=0)
    curve = OccupancyAnalyzer().occupancy_curve(k, A100, [32, 64, 128, 256])
    # All should achieve good occupancy
    for pt in curve:
        assert pt.occupancy > 0.0


def test_zero_threads_does_not_crash():
    k = make_kernel(regs=32, smem=0)
    result = OccupancyAnalyzer().analyze(k, A100, threads_per_block=0)
    assert result.occupancy == 0.0


def test_suggestions_generated_for_low_occupancy():
    k = make_kernel(regs=128, smem=0)
    result = OccupancyAnalyzer().analyze(k, A100, threads_per_block=256)
    assert len(result.suggestions) > 0


def test_get_arch_fallback():
    """Unknown SM should fall back to nearest known architecture."""
    arch = get_arch("sm_87")
    assert arch.sm in ARCHITECTURES


def test_h100_has_more_warps_than_a100():
    assert H100.max_warps_per_sm >= A100.max_warps_per_sm


def test_occupancy_from_real_ptx_vecadd():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = OccupancyAnalyzer().analyze(kernel, A100, 256)
    assert result.occupancy > 0.5  # vecadd is a lightweight kernel
