"""Tests for Triton code generation templates."""
from cudasage.codegen.triton_gen import TritonKernelGenerator, TritonOp, DType


def test_codegen_module_can_generate_all_ops_as_python_source() -> None:
    gen = TritonKernelGenerator(activation="relu", block_size=128)
    for op in TritonOp:
        result = gen.generate(op, DType.FLOAT16)
        # Validate generated source is syntactically valid Python.
        compile(result.source, f"<generated:{op.value}>", "exec")


def test_codegen_elementwise_activation_text_present() -> None:
    gen = TritonKernelGenerator(activation="silu", block_size=128)
    result = gen.generate(TritonOp.ELEMENTWISE, DType.FLOAT32)
    assert "silu" in result.source
    assert "elementwise_add_kernel" in result.source
