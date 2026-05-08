"""Integration tests for the CLI commands using typer's test runner.

Covers: analyze (text + json), diff, list-archs, error paths, and --output.
"""
import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from cudasage.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# list-archs
# ─────────────────────────────────────────────────────────────────────────────

def test_list_archs_exit_zero():
    result = runner.invoke(app, ["list-archs"])
    assert result.exit_code == 0


def test_list_archs_contains_sm_targets():
    result = runner.invoke(app, ["list-archs"])
    for sm in ("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"):
        assert sm in result.output


def test_list_archs_contains_gpu_names():
    result = runner.invoke(app, ["list-archs"])
    assert "A100" in result.output
    assert "H100" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# analyze — text output
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_vecadd_exit_zero():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "vecadd.ptx"), "--arch", "sm_80"])
    assert result.exit_code == 0


def test_analyze_vecadd_mentions_kernel_name():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "vecadd.ptx"), "--arch", "sm_80"])
    assert "vecadd" in result.output


def test_analyze_divergent_kernel_exit_zero():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "divergent_kernel.ptx"), "--arch", "sm_86"])
    assert result.exit_code == 0


def test_analyze_divergent_kernel_flags_divergence():
    result = runner.invoke(app, ["analyze", str(FIXTURES / "divergent_kernel.ptx"), "--arch", "sm_86"])
    # Should mention divergence
    assert "divergence" in result.output.lower() or "divergent" in result.output.lower()


def test_analyze_with_curve_flag():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--curve",
    ])
    assert result.exit_code == 0
    assert "Occupancy" in result.output


def test_analyze_file_not_found_exits_one():
    result = runner.invoke(app, ["analyze", "nonexistent.ptx"])
    assert result.exit_code == 1


def test_analyze_threads_guard_exits_one():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--threads", "0",
    ])
    assert result.exit_code == 1


def test_analyze_kernel_filter_match():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--kernel", "vecadd",
    ])
    assert result.exit_code == 0
    assert "vecadd" in result.output


def test_analyze_kernel_filter_no_match():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--kernel", "xyznotexist",
    ])
    assert result.exit_code == 0
    assert "No kernels matching" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# analyze — JSON output
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_json_format_parses():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1


def test_analyze_json_has_required_keys():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json",
    ])
    data = json.loads(result.output)
    report = data[0]
    for key in ("kernel", "sm_target", "overview", "occupancy", "divergence", "memory"):
        assert key in report, f"Missing key: {key}"


def test_analyze_json_occupancy_value_in_range():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json",
    ])
    data = json.loads(result.output)
    occ = data[0]["occupancy"]["value"]
    assert 0.0 <= occ <= 1.0


def test_analyze_json_divergent_kernel_has_sites():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "divergent_kernel.ptx"),
        "--arch", "sm_86", "--format", "json",
    ])
    data = json.loads(result.output)
    assert data[0]["divergence"]["has_divergence"] is True
    assert data[0]["divergence"]["site_count"] >= 1


def test_analyze_json_vecadd_no_spills():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json",
    ])
    data = json.loads(result.output)
    assert data[0]["memory"]["spill_ops"] == 0


def test_analyze_json_divergent_kernel_has_spills():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "divergent_kernel.ptx"),
        "--arch", "sm_86", "--format", "json",
    ])
    data = json.loads(result.output)
    assert data[0]["memory"]["spill_ops"] > 0


def test_analyze_json_with_curve():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json", "--curve",
    ])
    data = json.loads(result.output)
    curve = data[0]["occupancy_curve"]
    assert len(curve) == 10


def test_analyze_json_output_to_file(tmp_path):
    out_file = tmp_path / "report.json"
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "json",
        "--output", str(out_file),
    ])
    assert result.exit_code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert len(data) == 1
    assert data[0]["kernel"] == "vecadd"


def test_analyze_invalid_format_value_exits_nonzero():
    result = runner.invoke(app, [
        "analyze", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80", "--format", "yaml",
    ])
    assert result.exit_code != 0


# ─────────────────────────────────────────────────────────────────────────────
# diff command
# ─────────────────────────────────────────────────────────────────────────────

def test_diff_same_file_exit_zero():
    """Diffing a file against itself should produce NEUTRAL verdict."""
    result = runner.invoke(app, [
        "diff",
        str(FIXTURES / "vecadd.ptx"),
        str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80",
    ])
    assert result.exit_code == 0


def test_diff_same_file_shows_kernel():
    result = runner.invoke(app, [
        "diff",
        str(FIXTURES / "vecadd.ptx"),
        str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80",
    ])
    assert "vecadd" in result.output


def test_diff_no_common_kernels():
    """Diffing two files with no matching kernel names should warn."""
    result = runner.invoke(app, [
        "diff",
        str(FIXTURES / "vecadd.ptx"),
        str(FIXTURES / "divergent_kernel.ptx"),
        "--arch", "sm_80",
    ])
    assert result.exit_code == 0
    assert "No matching kernel names" in result.output


def test_diff_missing_baseline_exits_one():
    result = runner.invoke(app, [
        "diff", "missing.ptx", str(FIXTURES / "vecadd.ptx"),
        "--arch", "sm_80",
    ])
    assert result.exit_code == 1


def test_diff_missing_optimized_exits_one():
    result = runner.invoke(app, [
        "diff", str(FIXTURES / "vecadd.ptx"), "missing.ptx",
        "--arch", "sm_80",
    ])
    assert result.exit_code == 1


def test_diff_threads_guard_exits_one():
    result = runner.invoke(app, [
        "diff",
        str(FIXTURES / "vecadd.ptx"),
        str(FIXTURES / "vecadd.ptx"),
        "--threads", "0",
    ])
    assert result.exit_code == 1


def test_analyze_uses_env_default_arch(monkeypatch):
    monkeypatch.setenv("CUDA_SAGE_DEFAULT_ARCH", "sm_86")
    result = runner.invoke(app, ["analyze", str(FIXTURES / "vecadd.ptx")])
    assert result.exit_code == 0
    assert "sm_86" in result.output
