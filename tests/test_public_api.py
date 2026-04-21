"""Tests for the public cudasage package API and edge cases."""
import pytest
from cudasage import (
    PTXParser, KernelInfo, RegisterFile, Instruction,
    OccupancyAnalyzer, OccupancyResult, OccupancyCurvePoint,
    DivergenceAnalyzer, DivergenceResult, DivergenceSite,
    MemoryAnalyzer, MemoryResult, SpillWarning, BankConflictRisk,
    Architecture, ARCHITECTURES, get_arch,
    __version__,
)


# ─────────────────────────────────────────────────────────────────────────────
# Package meta
# ─────────────────────────────────────────────────────────────────────────────

def test_version_string():
    assert __version__ == "0.2.0"


def test_all_symbols_importable():
    """All public symbols must be importable from the top-level package."""
    for name in (
        "PTXParser", "KernelInfo", "RegisterFile",
        "OccupancyAnalyzer", "OccupancyResult",
        "DivergenceAnalyzer", "DivergenceResult",
        "MemoryAnalyzer", "MemoryResult",
        "Architecture", "ARCHITECTURES", "get_arch",
    ):
        import cudasage
        assert hasattr(cudasage, name), f"Missing export: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# PTXParser edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_empty_string():
    kernels = PTXParser().parse_string("")
    assert kernels == []


def test_parse_no_entry():
    ptx = ".version 7.0\n.target sm_80\n"
    kernels = PTXParser().parse_string(ptx)
    assert kernels == []


def test_parse_multiple_kernels():
    ptx = """
.version 7.0
.target sm_80
.entry kernel_a(.param .u32 n) {
    .reg .b32 %r<2>;
    ret;
}
.entry kernel_b(.param .u32 m) {
    .reg .b32 %r<4>;
    ret;
}
"""
    kernels = PTXParser().parse_string(ptx)
    assert len(kernels) == 2
    names = {k.name for k in kernels}
    assert "kernel_a" in names
    assert "kernel_b" in names


def test_register_file_physical_regs_no_regs():
    rf = RegisterFile()
    assert rf.physical_regs == 0


def test_register_file_pred_only():
    rf = RegisterFile(pred=8)
    # 8 pred → 1 pred slot
    assert rf.physical_regs == 1


def test_register_file_mixed():
    rf = RegisterFile(b32=10, f64=2)
    # 10 b32 + 2*2 f64 = 14
    assert rf.physical_regs == 14


def test_parse_no_sm_target_defaults_sm80():
    ptx = """
.version 7.0
.entry k(.param .u32 n) {
    ret;
}
"""
    kernel = PTXParser().parse_string(ptx)[0]
    assert kernel.sm_target == "sm_80"


def test_parse_shared_memory_size():
    ptx = """
.version 7.0
.target sm_80
.entry k(.param .u64 p) {
    .reg .b32 %r<2>;
    .shared .align 4 .b8 smem[4096];
    ret;
}
"""
    kernel = PTXParser().parse_string(ptx)[0]
    assert kernel.shared_mem_bytes == 4096
    assert len(kernel.shared_decls) == 1
    assert kernel.shared_decls[0].size_bytes == 4096


def test_parse_param_count():
    ptx = """
.version 7.0
.target sm_80
.visible .entry triple_param(
    .param .u64 a,
    .param .u64 b,
    .param .u32 n
) {
    .reg .b32 %r<2>;
    ret;
}
"""
    # param count parsed per separate .param line (standard PTX layout)
    kernel = PTXParser().parse_string(ptx)[0]
    assert kernel.param_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Architecture / get_arch edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_all_architectures_present():
    for sm in ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"):
        assert sm in ARCHITECTURES


def test_get_arch_exact_match():
    arch = get_arch("sm_80")
    assert arch.sm == "sm_80"


def test_get_arch_future_sm_returns_highest():
    arch = get_arch("sm_99")
    # Should return the highest known arch, not crash
    assert arch is not None
    assert arch.sm in ARCHITECTURES


def test_get_arch_nonsense_string():
    # Should not raise; falls back to sm_80
    arch = get_arch("banana")
    assert arch.sm == "sm_80"


def test_architecture_frozen():
    arch = ARCHITECTURES["sm_80"]
    with pytest.raises((AttributeError, TypeError)):
        arch.max_warps_per_sm = 999  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_occupancy_exceeds_max_threads_returns_zero():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.registers.b32 = 8
    arch = ARCHITECTURES["sm_80"]
    result = OccupancyAnalyzer().analyze(k, arch, threads_per_block=9999)
    assert result.occupancy == 0.0


def test_occupancy_single_thread_block():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.registers.b32 = 8
    arch = ARCHITECTURES["sm_80"]
    result = OccupancyAnalyzer().analyze(k, arch, threads_per_block=1)
    # 1 thread = 1 warp; should compute without crash
    assert 0.0 < result.occupancy <= 1.0


def test_occupancy_suggestions_register_limit():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.registers.b32 = 128  # very high register pressure
    arch = ARCHITECTURES["sm_80"]
    result = OccupancyAnalyzer().analyze(k, arch, threads_per_block=256)
    combined = " ".join(result.suggestions).lower()
    assert "register" in combined


def test_occupancy_curve_custom_sizes():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.registers.b32 = 16
    arch = ARCHITECTURES["sm_80"]
    sizes = [64, 128, 256]
    curve = OccupancyAnalyzer().occupancy_curve(k, arch, sizes)
    assert len(curve) == 3
    assert all(pt.threads_per_block in sizes for pt in curve)


# ─────────────────────────────────────────────────────────────────────────────
# Divergence edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_divergence_empty_kernel():
    k = KernelInfo(name="empty", sm_target="sm_80")
    result = DivergenceAnalyzer().analyze(k)
    assert not result.has_divergence
    assert result.tainted_regs == set()


def test_divergence_warpid_taints():
    """Branches on %warpid should also be tainted (warpid differs between warps
    but is constant within a warp — we conservatively flag it)."""
    ptx = """
.version 7.0
.target sm_80
.entry warpbranch(.param .u32 n) {
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    mov.u32 %r1, %warpid;
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra $done;
$done:
    ret;
}
"""
    kernel = PTXParser().parse_string(ptx)[0]
    result = DivergenceAnalyzer().analyze(kernel)
    # warpid is in _RE_TID_SRC — should be tainted
    assert len(result.tainted_regs) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Memory edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_memory_empty_kernel():
    k = KernelInfo(name="empty", sm_target="sm_80")
    result = MemoryAnalyzer().analyze(k)
    assert not result.has_spills
    assert not result.possible_missing_sync
    assert result.arithmetic_intensity_proxy == float("inf")


def test_memory_global_only_kernel():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.global_loads = 10
    k.global_stores = 2
    k.arithmetic = 5
    result = MemoryAnalyzer().analyze(k)
    assert result.global_load_count == 10
    assert result.global_store_count == 2
    # 5 / (10+2) ≈ 0.42
    assert abs(result.arithmetic_intensity_proxy - 5 / 12) < 0.01


def test_memory_spill_severity_critical():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.local_loads = 20  # > 10 threshold → critical
    result = MemoryAnalyzer().analyze(k)
    assert result.has_spills
    assert any(w.severity == "critical" for w in result.spill_warnings)


def test_memory_spill_severity_warning():
    k = KernelInfo(name="t", sm_target="sm_80")
    k.local_loads = 5  # ≤ 10 → warning
    result = MemoryAnalyzer().analyze(k)
    assert result.has_spills
    assert any(w.severity == "warning" for w in result.spill_warnings)
