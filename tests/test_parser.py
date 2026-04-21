"""Tests for the PTX parser."""
import pytest
from pathlib import Path
from cudasage.parsers.ptx_parser import PTXParser

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_vecadd_finds_one_kernel():
    kernels = PTXParser().parse_file(FIXTURES / "vecadd.ptx")
    assert len(kernels) == 1
    assert kernels[0].name == "vecadd"


def test_parse_vecadd_sm_target():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    assert kernel.sm_target == "sm_80"


def test_parse_vecadd_registers():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    assert kernel.registers.pred >= 2
    assert kernel.registers.b32 >= 8
    assert kernel.registers.b64 >= 12
    assert kernel.registers.f32 >= 4


def test_parse_vecadd_global_mem_ops():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    assert kernel.global_loads == 2   # a[tid] and b[tid]
    assert kernel.global_stores == 1  # c[tid]


def test_parse_vecadd_no_spills():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    assert kernel.local_loads == 0
    assert kernel.local_stores == 0


def test_parse_vecadd_no_shared_mem():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    assert kernel.shared_mem_bytes == 0


def test_parse_divergent_kernel_finds_one_kernel():
    kernels = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")
    assert len(kernels) == 1
    assert kernels[0].name == "divergent_kernel"


def test_parse_divergent_kernel_has_spills():
    kernel = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")[0]
    assert kernel.local_loads > 0
    assert kernel.local_stores > 0


def test_parse_divergent_kernel_has_special_fns():
    kernel = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")[0]
    assert kernel.special_fns >= 2  # sin and cos


def test_parse_from_string():
    ptx = """
.version 7.0
.target sm_90
.entry mykernel(.param .u32 p) {
    .reg .b32 %r<4>;
    mov.u32 %r1, %tid.x;
    ret;
}
"""
    kernels = PTXParser().parse_string(ptx)
    assert len(kernels) == 1
    assert kernels[0].name == "mykernel"
    assert kernels[0].sm_target == "sm_90"


def test_physical_regs_calculation():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    # Should count b32, b64 (×2), f32, pred (rounded up)
    phys = kernel.registers.physical_regs
    assert phys > 0
    assert phys <= 255  # hardware limit
