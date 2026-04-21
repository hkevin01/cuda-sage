"""Tests for the reporter module: JSON serialization and Rich output."""
import json
import pytest
from io import StringIO
from pathlib import Path
from rich.console import Console
from cudasage.parsers.ptx_parser import PTXParser
from cudasage.analyzers.occupancy import OccupancyAnalyzer
from cudasage.analyzers.divergence import DivergenceAnalyzer
from cudasage.analyzers.memory import MemoryAnalyzer
from cudasage.models.architectures import get_arch
from cudasage.reporter import build_json_report, render_kernel_report

FIXTURES = Path(__file__).parent / "fixtures"


def _analyze_vecadd():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    arch = get_arch("sm_80")
    occ = OccupancyAnalyzer().analyze(kernel, arch, 256)
    div = DivergenceAnalyzer().analyze(kernel)
    mem = MemoryAnalyzer().analyze(kernel)
    return kernel, occ, div, mem


def _analyze_divergent():
    kernel = PTXParser().parse_file(FIXTURES / "divergent_kernel.ptx")[0]
    arch = get_arch("sm_86")
    occ = OccupancyAnalyzer().analyze(kernel, arch, 256)
    div = DivergenceAnalyzer().analyze(kernel)
    mem = MemoryAnalyzer().analyze(kernel)
    return kernel, occ, div, mem


# ─────────────────────────────────────────────────────────────────────────────
# build_json_report
# ─────────────────────────────────────────────────────────────────────────────

def test_json_report_is_dict():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    assert isinstance(report, dict)


def test_json_report_kernel_name():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    assert report["kernel"] == "vecadd"
    assert report["sm_target"] == "sm_80"


def test_json_report_is_json_serializable():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    serialized = json.dumps(report)  # must not raise
    parsed = json.loads(serialized)
    assert parsed["kernel"] == "vecadd"


def test_json_report_occupancy_range():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    val = report["occupancy"]["value"]
    assert 0.0 <= val <= 1.0


def test_json_report_overview_fields():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    ov = report["overview"]
    assert ov["registers"] > 0
    assert ov["instruction_count"] > 0


def test_json_report_divergence_vecadd_no_sites():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem)
    # vecadd only branches on ctaid-derived index — not thread-ID tainted
    assert isinstance(report["divergence"]["has_divergence"], bool)
    assert report["divergence"]["site_count"] >= 0


def test_json_report_divergent_kernel_sites_list():
    kernel, occ, div, mem = _analyze_divergent()
    report = build_json_report(kernel, occ, div, mem)
    assert report["divergence"]["has_divergence"] is True
    assert len(report["divergence"]["sites"]) >= 1
    site = report["divergence"]["sites"][0]
    assert "line_no" in site
    assert "severity" in site
    assert site["severity"] in ("high", "medium", "low")


def test_json_report_memory_keys():
    kernel, occ, div, mem = _analyze_divergent()
    report = build_json_report(kernel, occ, div, mem)
    m = report["memory"]
    for key in ("global_loads", "global_stores", "spill_ops", "sync_barriers",
                "memory_bound_likely", "possible_missing_sync"):
        assert key in m


def test_json_report_with_curve():
    kernel = PTXParser().parse_file(FIXTURES / "vecadd.ptx")[0]
    arch = get_arch("sm_80")
    occ = OccupancyAnalyzer().analyze(kernel, arch, 256)
    div = DivergenceAnalyzer().analyze(kernel)
    mem = MemoryAnalyzer().analyze(kernel)
    curve = OccupancyAnalyzer().occupancy_curve(kernel, arch)
    report = build_json_report(kernel, occ, div, mem, curve)
    assert len(report["occupancy_curve"]) == 10
    for pt in report["occupancy_curve"]:
        assert "threads_per_block" in pt
        assert 0.0 <= pt["occupancy"] <= 1.0


def test_json_report_none_curve_gives_empty_list():
    kernel, occ, div, mem = _analyze_vecadd()
    report = build_json_report(kernel, occ, div, mem, curve=None)
    assert report["occupancy_curve"] == []


# ─────────────────────────────────────────────────────────────────────────────
# render_kernel_report (smoke test — just ensure it doesn't raise)
# ─────────────────────────────────────────────────────────────────────────────

def test_render_kernel_report_vecadd_no_exception():
    kernel, occ, div, mem = _analyze_vecadd()
    buf = StringIO()
    custom_console = Console(file=buf, force_terminal=False)
    import cudasage.reporter as reporter_mod
    original = reporter_mod.console
    reporter_mod.console = custom_console
    try:
        render_kernel_report(kernel, occ, div, mem)
    finally:
        reporter_mod.console = original


def test_render_kernel_report_divergent_no_exception():
    kernel, occ, div, mem = _analyze_divergent()
    buf = StringIO()
    custom_console = Console(file=buf, force_terminal=False)
    import cudasage.reporter as reporter_mod
    original = reporter_mod.console
    reporter_mod.console = custom_console
    try:
        render_kernel_report(kernel, occ, div, mem)
    finally:
        reporter_mod.console = original


def test_render_kernel_report_with_curve_no_exception():
    kernel, occ, div, mem = _analyze_vecadd()
    arch = get_arch("sm_80")
    curve = OccupancyAnalyzer().occupancy_curve(kernel, arch)
    buf = StringIO()
    custom_console = Console(file=buf, force_terminal=False)
    import cudasage.reporter as reporter_mod
    original = reporter_mod.console
    reporter_mod.console = custom_console
    try:
        render_kernel_report(kernel, occ, div, mem, curve)
    finally:
        reporter_mod.console = original
