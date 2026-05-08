"""cuda-sage CLI entry point.

Usage:
  cuda-sage analyze kernel.ptx [--arch sm_80] [--threads 256] [--curve]
  cuda-sage diff baseline.ptx optimized.ptx [--arch sm_80]
  cuda-sage list-archs
"""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich import print as rprint
from rich.table import Table

from . import __version__
from .analyzers.divergence import DivergenceAnalyzer
from .analyzers.memory import MemoryAnalyzer
from .analyzers.occupancy import OccupancyAnalyzer
from .models.architectures import ARCHITECTURES, get_arch
from .parsers.ptx_parser import PTXParser
from .reporter import build_json_report, console, render_kernel_report

app = typer.Typer(
    name="cuda-sage",
    help="CUDA/PTX Static Analysis & Guidance Engine - no GPU required.",
    add_completion=False,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _env_default_arch() -> str:
    """Return configured default architecture from env, with safe fallback."""
    value = os.getenv("CUDA_SAGE_DEFAULT_ARCH", "sm_80").strip()
    return value or "sm_80"


def _env_default_threads() -> int:
    """Return configured default threads/block from env, with safe fallback."""
    raw = os.getenv("CUDA_SAGE_DEFAULT_THREADS", "256").strip()
    try:
        value = int(raw)
    except ValueError:
        return 256
    return value if value > 0 else 256


def _init_analyzers() -> tuple[OccupancyAnalyzer, DivergenceAnalyzer, MemoryAnalyzer]:
    """Initialize analyzers in one place to keep command setup consistent."""
    return OccupancyAnalyzer(), DivergenceAnalyzer(), MemoryAnalyzer()


def _version_callback(value: bool) -> None:
    if value:
        rprint(f"cuda-sage v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


@app.command()
def analyze(
    ptx_file: Path = typer.Argument(..., help="Path to .ptx file to analyze"),
    arch: str = typer.Option(_env_default_arch(), "--arch", "-a", help="Target SM architecture (e.g. sm_80, sm_90)"),
    threads: int = typer.Option(_env_default_threads(), "--threads", "-t", help="Assumed threads per block for occupancy"),
    curve: bool = typer.Option(False, "--curve", "-c", help="Show occupancy curve across block sizes"),
    kernel_filter: Optional[str] = typer.Option(None, "--kernel", "-k", help="Analyze only kernels matching this name"),
    fmt: OutputFormat = typer.Option(OutputFormat.text, "--format", "-f", help="Output format: text or json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write results to this file instead of stdout"),
) -> None:
    """Analyze a PTX file for occupancy, warp divergence, and memory issues."""
    if not ptx_file.exists() or not ptx_file.is_file():
        rprint(f"[red]Error:[/] File not found: {ptx_file}")
        raise typer.Exit(1)
    if threads <= 0:
        rprint(f"[red]Error:[/] --threads must be positive (got {threads})")
        raise typer.Exit(1)

    parser = PTXParser()
    kernels = parser.parse_file(ptx_file)

    if not kernels:
        rprint(f"[yellow]No kernel entries (.entry) found in {ptx_file}[/]")
        raise typer.Exit(0)

    arch_spec = get_arch(arch)
    occ_analyzer, div_analyzer, mem_analyzer = _init_analyzers()

    filtered = [k for k in kernels if not kernel_filter or kernel_filter in k.name]
    if not filtered:
        rprint(f"[yellow]No kernels matching '{kernel_filter}'[/]")
        raise typer.Exit(0)

    if fmt == OutputFormat.json:
        results = []
        for kernel in filtered:
            occ_result = occ_analyzer.analyze(kernel, arch_spec, threads)
            occ_curve = occ_analyzer.occupancy_curve(kernel, arch_spec) if curve else None
            div_result = div_analyzer.analyze(kernel)
            mem_result = mem_analyzer.analyze(kernel)
            results.append(build_json_report(kernel, occ_result, div_result, mem_result, occ_curve))
        payload = json.dumps(results, indent=2)
        if output:
            if output.parent and not output.parent.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            rprint(f"[green]Results written to {output}[/]")
        else:
            print(payload)
        return

    rprint(
        f"\n[bold]cuda-sage[/] [dim]v{__version__}[/] - analyzing [cyan]{ptx_file}[/] "
        f"against [bold]{arch_spec.name}[/] ([dim]{arch_spec.sm}[/])\n"
    )

    for kernel in filtered:
        occ_result = occ_analyzer.analyze(kernel, arch_spec, threads)
        occ_curve = occ_analyzer.occupancy_curve(kernel, arch_spec) if curve else None
        div_result = div_analyzer.analyze(kernel)
        mem_result = mem_analyzer.analyze(kernel)
        render_kernel_report(kernel, occ_result, div_result, mem_result, occ_curve)


@app.command()
def diff(
    baseline: Path = typer.Argument(..., help="Baseline PTX file"),
    optimized: Path = typer.Argument(..., help="Optimized PTX file to compare against baseline"),
    arch: str = typer.Option(_env_default_arch(), "--arch", "-a"),
    threads: int = typer.Option(_env_default_threads(), "--threads", "-t"),
) -> None:
    """Compare two PTX files and report performance regressions or improvements."""
    for f in (baseline, optimized):
        if not f.exists() or not f.is_file():
            rprint(f"[red]Error:[/] File not found: {f}")
            raise typer.Exit(1)
    if threads <= 0:
        rprint(f"[red]Error:[/] --threads must be positive (got {threads})")
        raise typer.Exit(1)

    parser = PTXParser()
    arch_spec = get_arch(arch)
    occ_analyzer, div_analyzer, mem_analyzer = _init_analyzers()

    base_kernels = {k.name: k for k in parser.parse_file(baseline)}
    opt_kernels = {k.name: k for k in parser.parse_file(optimized)}

    common = set(base_kernels) & set(opt_kernels)
    if not common:
        rprint("[yellow]No matching kernel names found between the two files.[/]")
        raise typer.Exit(0)

    rprint(f"\n[bold]cuda-sage diff[/] - [cyan]{baseline.name}[/] -> [cyan]{optimized.name}[/]\n")

    table = Table(title=f"Performance Delta ({arch_spec.sm})", box=box.ROUNDED)
    table.add_column("Kernel", style="bold")
    table.add_column("Occupancy D", justify="right")
    table.add_column("Regs D", justify="right")
    table.add_column("Spills D", justify="right")
    table.add_column("Divergence sites D", justify="right")
    table.add_column("Verdict")

    for name in sorted(common):
        bk, ok = base_kernels[name], opt_kernels[name]
        b_occ = occ_analyzer.analyze(bk, arch_spec, threads)
        o_occ = occ_analyzer.analyze(ok, arch_spec, threads)
        b_div = div_analyzer.analyze(bk)
        o_div = div_analyzer.analyze(ok)
        b_mem = mem_analyzer.analyze(bk)
        o_mem = mem_analyzer.analyze(ok)

        d_occ = o_occ.occupancy - b_occ.occupancy
        d_regs = ok.registers.physical_regs - bk.registers.physical_regs
        d_spill = sum(w.count for w in o_mem.spill_warnings) - sum(w.count for w in b_mem.spill_warnings)
        d_div = len(o_div.sites) - len(b_div.sites)

        occ_str = f"[{'green' if d_occ >= 0 else 'red'}]{d_occ:+.1%}[/]"
        regs_str = f"[{'red' if d_regs > 0 else 'green' if d_regs < 0 else 'dim'}]{d_regs:+d}[/]"
        spill_str = f"[{'red' if d_spill > 0 else 'green' if d_spill < 0 else 'dim'}]{d_spill:+d}[/]"
        div_str = f"[{'red' if d_div > 0 else 'green' if d_div < 0 else 'dim'}]{d_div:+d}[/]"

        is_regression = (d_occ < -0.05) or (d_regs > 8) or (d_spill > 0) or (d_div > 0)
        is_improvement = (d_occ > 0.05) or (d_regs < 0 and d_spill <= 0)
        verdict = (
            "[red]REGRESSION[/]"
            if is_regression
            else ("[green]IMPROVED[/]" if is_improvement else "[dim]NEUTRAL[/]")
        )

        table.add_row(name, occ_str, regs_str, spill_str, div_str, verdict)

    console.print(table)


@app.command(name="list-archs")
def list_archs() -> None:
    """List all supported GPU architectures."""
    table = Table(title="Supported Architectures", box=box.ROUNDED)
    table.add_column("SM Target", style="bold cyan")
    table.add_column("GPU Name")
    table.add_column("Max Warps/SM", justify="right")
    table.add_column("Max Threads/SM", justify="right")
    table.add_column("Regs/SM", justify="right")
    table.add_column("Shared Mem/SM")

    for sm, arch in sorted(ARCHITECTURES.items()):
        table.add_row(
            sm,
            arch.name,
            str(arch.max_warps_per_sm),
            str(arch.max_threads_per_sm),
            str(arch.regs_per_sm),
            f"{arch.smem_per_sm_bytes // 1024} KB",
        )
    console.print(table)


if __name__ == "__main__":
    app()