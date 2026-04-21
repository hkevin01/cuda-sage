"""Shared pytest fixtures for cuda-sage test suite."""
from __future__ import annotations
import pytest
from pathlib import Path
from cudasage.parsers.ptx_parser import PTXParser
from cudasage.models.architectures import get_arch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def parser():
    return PTXParser()


@pytest.fixture(scope="session")
def vecadd_kernel(parser):
    return parser.parse_file(FIXTURES / "vecadd.ptx")[0]


@pytest.fixture(scope="session")
def divergent_kernel(parser):
    return parser.parse_file(FIXTURES / "divergent_kernel.ptx")[0]


@pytest.fixture(scope="session")
def matmul_kernel(parser):
    return parser.parse_file(FIXTURES / "matmul.ptx")[0]


@pytest.fixture(scope="session")
def reduction_kernel(parser):
    return parser.parse_file(FIXTURES / "reduction.ptx")[0]


@pytest.fixture(scope="session")
def arch_sm80():
    return get_arch("sm_80")


@pytest.fixture(scope="session")
def arch_sm86():
    return get_arch("sm_86")


@pytest.fixture(scope="session")
def arch_sm90():
    return get_arch("sm_90")
