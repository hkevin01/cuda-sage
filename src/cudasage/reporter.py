"""Rich terminal reporter.

Assembles occupancy, divergence, and memory analysis results into a
well-structured, colorized terminal report using the Rich library.
"""
from __future__ import annotations
import json
import math
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from .parsers.ptx_parser import KernelInfo
from .analyzers.occupancy import OccupancyResult, OccupancyCurvePoint
from .analyzers.divergence import DivergenceResult
from .analyzers.memory import MemoryResult

console = Console()

_SEVERITY_COLORS = {"high": "red", "medium": "yellow", "low": "cyan", "critical": "bold red"}
_OCC_COLORS = [(0.0, "red"), (0.25, "yellow"), (0.5, "green"), (0.75, "bright_green")]


def _occ_color(occ: float) -> str:
    if occ != occ:  # NaN guard
        return "red"
    color = "red"
    for threshold, c in _OCC_COLORS:
        if occ >= threshold:
            color = c
    return color


def _occ_bar(occ: float, width: int = 30) -> str:
    if width <= 0:
        return ""
    occ = max(0.0, min(1.0, occ))
    filled = round(occ * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Function:       render_kernel_report
# Purpose:        Print a complete colorized analysis report for one kernel.
# Inputs:         kernel: KernelInfo — parsed metadata
#                 occ: OccupancyResult — occupancy calculation
#                 div: DivergenceResult — divergence sites
#                 mem: MemoryResult — memory pattern findings
#                 curve: list[OccupancyCurvePoint] | None — optional curve table
# Side Effects:   Writes to stdout via Rich Console.
# ─────────────────────────────────────────────────────────────────────────────
def render_kernel_report(
    kernel: KernelInfo,
    occ: OccupancyResult,
    div: DivergenceResult,
    mem: MemoryResult,
    curve: list[OccupancyCurvePoint] | None = None,
) -> None:
    """Print a complete analysis report for one kernel to the terminal."""
    console.rule(f"[bold cyan]Kernel: {kernel.name}[/]  [dim]({kernel.sm_target})[/]")

    # ── Section 1: Overview ───────────────────────────────────────────────
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold", no_wrap=True)
    overview.add_column()
    overview.add_row("PTX target",    kernel.sm_target)
    overview.add_row("Parameters",    str(kernel.param_count))
    overview.add_row("Registers",     f"{kernel.registers.physical_regs} physical ({kernel.registers.b32 + kernel.registers.f32} ×32b, {kernel.registers.b64 + kernel.registers.f64} ×64b, {kernel.registers.pred} pred)")
    overview.add_row("Shared memory", f"{kernel.shared_mem_bytes:,} bytes ({kernel.shared_mem_bytes / 1024:.2f} KB)" if kernel.shared_mem_bytes else "none")
    overview.add_row("Instructions",  str(len(kernel.instructions)))
    console.print(Panel(overview, title="[bold]Overview[/]", border_style="dim"))

    # ── Section 2: Occupancy ──────────────────────────────────────────────
    occ_color = _occ_color(occ.occupancy)
    occ_bar = _occ_bar(occ.occupancy)
    occ_text = Text()
    occ_text.append(f"  {occ.occupancy:.1%}  ", style=f"bold {occ_color}")
    occ_text.append(occ_bar, style=occ_color)
    occ_text.append(f"  [{occ.active_warps}/{occ.active_blocks * math.ceil(occ.threads_per_block / 32) if occ.active_blocks else 0} warps, {occ.active_blocks} blocks]", style="dim")

    occ_detail = Table.grid(padding=(0, 2))
    occ_detail.add_column(style="bold", no_wrap=True)
    occ_detail.add_column()
    occ_detail.add_row("Threads/block",       str(occ.threads_per_block))
    occ_detail.add_row("Active blocks/SM",    str(occ.active_blocks))
    occ_detail.add_row("Limiting factor",     f"[yellow]{occ.limiting_factor}[/]")
    occ_detail.add_row("Blocks (thread lim)", str(occ.blocks_by_threads))
    occ_detail.add_row("Blocks (reg lim)",    str(occ.blocks_by_regs))
    occ_detail.add_row("Blocks (smem lim)",   str(occ.blocks_by_smem))
    occ_detail.add_row("Blocks (HW lim)",     str(occ.blocks_by_hw_limit))

    console.print(Panel(
        Text.assemble(occ_text, "\n\n", "") if occ.occupancy < 0.5 else occ_text,
        title="[bold]Occupancy[/]",
        border_style="yellow" if occ.occupancy < 0.5 else "green",
    ))
    console.print(occ_detail)

    # Occupancy curve
    if curve:
        console.print("\n  [bold]Occupancy vs Block Size:[/]")
        curve_table = Table(box=box.SIMPLE, padding=(0, 1))
        curve_table.add_column("Threads/Block", style="dim", width=14)
        curve_table.add_column("Occupancy", width=8)
        curve_table.add_column("Bar", width=32)
        curve_table.add_column("Bottleneck", style="dim")
        for pt in curve:
            c = _occ_color(pt.occupancy)
            curve_table.add_row(
                str(pt.threads_per_block),
                f"[{c}]{pt.occupancy:.1%}[/]",
                f"[{c}]{_occ_bar(pt.occupancy, 24)}[/]",
                pt.limiting_factor,
            )
        console.print(curve_table)

    _print_suggestions(occ.suggestions, "Occupancy Suggestions")

    # ── Section 3: Divergence ─────────────────────────────────────────────
    div_color = "red" if div.has_divergence else "green"
    div_status = f"[{div_color}]{'⚠ ' + str(len(div.sites)) + ' divergence site(s) detected' if div.has_divergence else '✓ No thread-ID-driven divergence detected'}[/]"
    console.print(Panel(div_status, title="[bold]Warp Divergence[/]", border_style=div_color))

    if div.sites:
        div_table = Table(box=box.SIMPLE, padding=(0, 1))
        div_table.add_column("Line", style="dim", width=6)
        div_table.add_column("Predicate", width=10)
        div_table.add_column("Severity", width=8)
        div_table.add_column("Reason")
        for site in div.sites:
            sc = _SEVERITY_COLORS.get(site.severity, "white")
            div_table.add_row(
                str(site.line_no),
                site.predicate_reg,
                f"[{sc}]{site.severity.upper()}[/]",
                site.reason,
            )
        console.print(div_table)

    _print_suggestions(div.suggestions, "Divergence Suggestions")

    # ── Section 4: Memory ─────────────────────────────────────────────────
    mem_issues = len(mem.spill_warnings) + len(mem.bank_conflict_risks) + (1 if mem.possible_missing_sync else 0)
    mem_color = "red" if mem_issues > 0 else "green"

    mem_metrics = Table.grid(padding=(0, 2))
    mem_metrics.add_column(style="bold", no_wrap=True)
    mem_metrics.add_column()
    mem_metrics.add_row("Global loads",      str(mem.global_load_count))
    mem_metrics.add_row("Global stores",     str(mem.global_store_count))
    mem_metrics.add_row("Shared loads",      str(mem.shared_load_count))
    mem_metrics.add_row("Shared stores",     str(mem.shared_store_count))
    mem_metrics.add_row("Local (spill) ops", f"[{'red' if mem.has_spills else 'dim'}]{sum(w.count for w in mem.spill_warnings)}[/]")
    mem_metrics.add_row("Sync barriers",     str(mem.sync_barriers))
    ai = mem.arithmetic_intensity_proxy
    ai_str = f"{ai:.1f}" if ai != float('inf') else "∞"
    mem_metrics.add_row("Arith intensity",   f"{ai_str} ops/global-mem-op")
    mem_metrics.add_row("Bound estimate",    "[yellow]memory-bound[/]" if mem.memory_bound_likely else "[cyan]compute-bound[/]")

    console.print(Panel(mem_metrics, title="[bold]Memory Access[/]", border_style=mem_color))

    if mem.possible_missing_sync:
        console.print("[bold red]  ⚠ Possible missing __syncthreads() after shared memory write![/]")

    _print_suggestions(mem.suggestions, "Memory Suggestions")
    console.rule()


def _print_suggestions(suggestions: list[str], title: str) -> None:
    if not suggestions:
        return
    console.print(f"\n  [bold yellow]{title}:[/]")
    for i, s in enumerate(suggestions, 1):
        console.print(f"  [dim]{i}.[/] {s}")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Function:       build_json_report
# Purpose:        Serialize analysis results for one kernel to a JSON-serializable
#                 dict suitable for machine consumption (CI, toolchains, APIs).
# Inputs:         kernel: KernelInfo
#                 occ: OccupancyResult
#                 div: DivergenceResult
#                 mem: MemoryResult
#                 curve: list[OccupancyCurvePoint] | None
# Outputs:        dict — JSON-safe structure with all analysis fields
# Side Effects:   None (pure function).
# ─────────────────────────────────────────────────────────────────────────────
def build_json_report(
    kernel: KernelInfo,
    occ: OccupancyResult,
    div: DivergenceResult,
    mem: MemoryResult,
    curve: list[OccupancyCurvePoint] | None = None,
) -> dict:
    """Build a JSON-serializable dict for one kernel's full analysis."""
    return {
        "kernel": kernel.name,
        "sm_target": kernel.sm_target,
        "overview": {
            "registers": kernel.registers.physical_regs,
            "shared_mem_bytes": kernel.shared_mem_bytes,
            "instruction_count": len(kernel.instructions),
            "param_count": kernel.param_count,
        },
        "occupancy": {
            "value": round(occ.occupancy, 4),
            "percent": round(occ.occupancy * 100, 1),
            "active_blocks": occ.active_blocks,
            "active_warps": occ.active_warps,
            "threads_per_block": occ.threads_per_block,
            "limiting_factor": occ.limiting_factor,
            "blocks_by_threads": occ.blocks_by_threads,
            "blocks_by_regs": occ.blocks_by_regs,
            "blocks_by_smem": occ.blocks_by_smem,
            "blocks_by_hw_limit": occ.blocks_by_hw_limit,
            "suggestions": occ.suggestions,
        },
        "divergence": {
            "has_divergence": div.has_divergence,
            "site_count": len(div.sites),
            "high_severity_count": div.high_severity_count,
            "tainted_reg_count": len(div.tainted_regs),
            "sites": [
                {
                    "line_no": s.line_no,
                    "predicate": s.predicate_reg,
                    "severity": s.severity,
                    "reason": s.reason,
                    "line_text": s.line_text,
                }
                for s in div.sites
            ],
            "suggestions": div.suggestions,
        },
        "memory": {
            "global_loads": mem.global_load_count,
            "global_stores": mem.global_store_count,
            "shared_loads": mem.shared_load_count,
            "shared_stores": mem.shared_store_count,
            "spill_ops": sum(w.count for w in mem.spill_warnings),
            "sync_barriers": mem.sync_barriers,
            "arithmetic_intensity_proxy": (
                round(mem.arithmetic_intensity_proxy, 2)
                if mem.arithmetic_intensity_proxy != float("inf") else None
            ),
            "memory_bound_likely": mem.memory_bound_likely,
            "possible_missing_sync": mem.possible_missing_sync,
            "spill_warnings": [
                {"kind": w.kind, "count": w.count, "severity": w.severity}
                for w in mem.spill_warnings
            ],
            "bank_conflict_risks": [
                {"risk_level": r.risk_level, "stride": r.stride_hint, "description": r.description}
                for r in mem.bank_conflict_risks
            ],
            "suggestions": mem.suggestions,
        },
        "occupancy_curve": [
            {"threads_per_block": pt.threads_per_block,
             "occupancy": round(pt.occupancy, 4),
             "limiting_factor": pt.limiting_factor}
            for pt in (curve or [])
        ],
    }
