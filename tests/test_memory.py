"""Tests for memory access pattern analyzer."""
import pytest
from pathlib import Path
from cudasage.parsers.ptx_parser import PTXParser
from cudasage.analyzers.memory import MemoryAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_vecadd_no_spills():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert not result.has_spills


def test_vecadd_global_load_count():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert result.global_load_count == 2
    assert result.global_store_count == 1


def test_vecadd_no_missing_sync():
    """vecadd has no shared memory writes, so no sync warning expected."""
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert not result.possible_missing_sync


def test_divergent_kernel_has_spills():
    kernel = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert result.has_spills
    total_spills = sum(w.count for w in result.spill_warnings)
    assert total_spills > 0


def test_divergent_kernel_suggestions_mention_spills():
    kernel = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    combined = " ".join(result.suggestions).lower()
    assert "spill" in combined or "local memory" in combined


def test_arithmetic_intensity_proxy_vecadd():
    """vecadd: 1 add + some mov/cvt/mul ops vs 3 global mem ops."""
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    # Should be low (memory-bound style kernel)
    assert result.arithmetic_intensity_proxy < 10


def test_memory_bound_detection_vecadd():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert isinstance(result.memory_bound_likely, bool)


def test_missing_sync_detected():
    """Kernel with shared mem store but no bar.sync should warn."""
    ptx = """
.version 7.0
.target sm_80
.entry smem_no_sync(.param .u64 p_out) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<2>;
    .shared .align 4 .b8 buf[128];
    mov.u32 %r1, %tid.x;
    cvta.to.shared.u64 %rd1, buf;
    st.shared.f32 [%rd1], %f1;
    ret;
}
"""
    from cudasage.parsers.ptx_parser import PTXParser
    kernel = PTXParser().parse_string(ptx)[0]
    result = MemoryAnalyzer().analyze(kernel)
    assert result.possible_missing_sync
