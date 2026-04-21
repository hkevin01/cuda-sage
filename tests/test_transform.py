"""Tests for CUDA C source-to-source transformer."""
import pytest
from pathlib import Path
from cudasage.transform import CUDASourceTransformer, TransformResult

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


# ── T1: __launch_bounds__ ─────────────────────────────────────────────────────

def test_launch_bounds_injected_when_missing():
    src = "__global__ void mykernel(float* a) { }"
    result = CUDASourceTransformer().transform(src)
    assert "__launch_bounds__" in result.transformed_source


def test_launch_bounds_not_duplicated():
    src = "__global__ __launch_bounds__(256) void mykernel(float* a) { }"
    result = CUDASourceTransformer().transform(src)
    assert result.transformed_source.count("__launch_bounds__") == 1


def test_launch_bounds_uses_configured_block_size():
    src = "__global__ void mykernel(float* a) { }"
    result = CUDASourceTransformer(default_block_size=128).transform(src)
    assert "__launch_bounds__(128)" in result.transformed_source


def test_launch_bounds_reported_as_high_impact():
    src = "__global__ void mykernel(float* a) { }"
    result = CUDASourceTransformer().transform(src)
    lb_transforms = [t for t in result.transforms if t.name == "__launch_bounds__"]
    assert lb_transforms
    assert all(t.impact == "high" for t in lb_transforms)


def test_launch_bounds_applied_to_vecadd():
    src = load("vecadd.cu")
    result = CUDASourceTransformer().transform(src)
    assert "__launch_bounds__" in result.transformed_source
    assert result.has_changes


# ── T2: __restrict__ ──────────────────────────────────────────────────────────

def test_restrict_added_to_pointer_params():
    src = "__global__ void k(float* a, float* b, float* out, int n) { }"
    result = CUDASourceTransformer().transform(src)
    assert "__restrict__" in result.transformed_source


def test_restrict_not_double_added():
    src = "__global__ void k(float* __restrict__ a) { }"
    result = CUDASourceTransformer().transform(src)
    assert result.transformed_source.count("__restrict__") == 1


def test_restrict_applied_to_vecadd():
    src = load("vecadd.cu")
    result = CUDASourceTransformer().transform(src)
    assert "__restrict__" in result.transformed_source


def test_restrict_category_is_memory():
    src = "__global__ void k(float* a) { }"
    result = CUDASourceTransformer().transform(src)
    restrict_ts = [t for t in result.transforms if t.name == "__restrict__"]
    assert all(t.category == "memory" for t in restrict_ts)


# ── T3: Shared memory bank conflict padding ───────────────────────────────────

def test_shared_mem_padded_when_mult_of_32():
    src = "__global__ void k() { __shared__ float tile[32]; }"
    result = CUDASourceTransformer().transform(src)
    assert "tile[33]" in result.transformed_source


def test_shared_mem_not_padded_when_not_mult_of_32():
    src = "__global__ void k() { __shared__ float tile[31]; }"
    result = CUDASourceTransformer().transform(src)
    assert "tile[31]" in result.transformed_source   # unchanged


def test_shared_mem_padded_in_matmul():
    # matmul has TILE_SIZE=16 but shared arrays use [TILE_SIZE][TILE_SIZE]
    # After macro substitution the regex won't see the literal 16 in arrays
    # so this tests that 16×16 arrays with constant 16 ARE padded
    src = """__global__ void k() {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
}"""
    result = CUDASourceTransformer().transform(src)
    assert "17" in result.transformed_source


def test_shared_mem_transform_is_high_impact():
    src = "__global__ void k() { __shared__ float tile[32]; }"
    result = CUDASourceTransformer().transform(src)
    pad_ts = [t for t in result.transforms if t.name == "shared_mem_padding"]
    assert pad_ts
    assert all(t.impact == "high" for t in pad_ts)


# ── T4: #pragma unroll ────────────────────────────────────────────────────────

def test_pragma_unroll_inserted_for_small_bound():
    src = "__global__ void k() {\n    for (int i = 0; i < 8; i++) {}\n}"
    result = CUDASourceTransformer().transform(src)
    assert "#pragma unroll" in result.transformed_source


def test_pragma_unroll_not_inserted_for_large_bound():
    src = "__global__ void k() {\n    for (int i = 0; i < 1024; i++) {}\n}"
    result = CUDASourceTransformer(max_unroll_count=16).transform(src)
    assert "#pragma unroll" not in result.transformed_source


def test_pragma_unroll_not_duplicated():
    src = "__global__ void k() {\n    #pragma unroll 8\n    for (int i = 0; i < 8; i++) {}\n}"
    result = CUDASourceTransformer().transform(src)
    assert result.transformed_source.count("#pragma unroll") == 1


# ── T5: Divergence annotation ─────────────────────────────────────────────────

def test_divergence_hint_injected_for_tid_mod():
    src = "__global__ void k() {\n    if (threadIdx.x % 2 == 0) {}\n}"
    result = CUDASourceTransformer().transform(src)
    assert "PERF" in result.transformed_source or "ballot_sync" in result.transformed_source


def test_divergence_transform_is_high_impact():
    src = "__global__ void k() {\n    if (threadIdx.x % 2 == 0) {}\n}"
    result = CUDASourceTransformer().transform(src)
    div_ts = [t for t in result.transforms if t.name == "divergence_hint"]
    assert div_ts
    assert all(t.impact == "high" for t in div_ts)


def test_clean_kernel_has_no_divergence_hints():
    src = "__global__ void k(float* a) { a[threadIdx.x] = 1.0f; }"
    result = CUDASourceTransformer().transform(src)
    div_ts = [t for t in result.transforms if t.name == "divergence_hint"]
    assert not div_ts


# ── Integration: full transform pipeline ─────────────────────────────────────

def test_transform_result_summary_lines():
    src = load("vecadd.cu")
    result = CUDASourceTransformer().transform(src)
    assert len(result.summary()) >= 2   # at least launch_bounds + restrict


def test_high_impact_count_vecadd():
    src = load("vecadd.cu")
    result = CUDASourceTransformer().transform(src)
    assert result.high_impact_count >= 1   # at least __launch_bounds__


def test_transformed_source_is_string():
    src = load("vecadd.cu")
    result = CUDASourceTransformer().transform(src)
    assert isinstance(result.transformed_source, str)
    assert len(result.transformed_source) > len(src) - 100  # not much shorter
