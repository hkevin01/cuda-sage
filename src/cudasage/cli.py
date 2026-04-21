"""cuda-sage CLI entry point.

Usage:
  cuda-sage analyze kernel.ptx [--arch sm_80] [--threads 256] [--curve]
  cuda-sage diff baseline.ptx optimized.ptx [--arch sm_80]
  cuda-sage list-archs
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.table import Table
from rich import box
from .parsers.ptx_parser import PTXParser
from .analyzers.occupancy import OccupancyAnalyzer
from .analyzers.divergence import DivergenceAnalyzer
from .analyzers.memory import MemoryAnalyzer
from .models.architectures import ARCHITECTURES, get_arch
from .reporter import render_kernel_report, build_json_report, console
from . import __version__

app = typer.Typer(
    name="cuda-sage",
    help="CUDA/PTX Static Analysis & Guidance Engine — no GPU required.",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        rprint(f"cuda-sage v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Command:        analyze
# Purpose:        Run full static analysis on a PTX file and report results.
# Inputs:         ptx_file — path to .ptx source
#                 arch — SM target string (default: sm_80)
#                 threads — assumed threads/block for occupancy (default: 256)
#                 curve — if True, include occupancy curve table
#                 kernel_filter — optional substring filter on kernel names
#                 fmt — output format: "text" (Rich terminal) or "json"
#                 output — optional file path to write results to
# Side Effects:   Writes to stdout or to output file.
#                 Exit code 1 on file-not-found or no kernels found.
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def analyze(
    ptx_file: Path = typer.Argument(..., help="Path to .ptx file to analyze"),
    arch: str = typer.Option("sm_80", "--arch", "-a", help="Target SM architecture (e.g. sm_80, sm_90)"),
    threads: int = typer.Option(256, "--threads", "-t", help="Assumed threads per block for occupancy"),
    curve: bool = typer.Option(False, "--curve", "-c", help="Show occupancy curve across block sizes"),
    kernel_filter: Optional[str] = typer.Option(None, "--kernel", "-k", help="Analyze only kernels matching this name"),
    fmt: str = typer.Option("text", "--format", "-f", help="Output format: text or json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write results to this file instead of stdout"),
) -> None:
    """Analyze a PTX file for occupancy, warp divergence, and memory issues."""
    if not ptx_file.exists():
        rprint(f"[red]Error:[/] File not found: {ptx_file}")
        raise typer.Exit(1)

    parser = PTXParser()
    kernels = parser.parse_file(ptx_file)

    if not kernels:
        rprint(f"[yellow]No kernel entries (.entry) found in {ptx_file}[/]")
        raise typer.Exit(0)

    arch_spec = get_arch(arch)
    occ_analyzer = OccupancyAnalyzer()
    div_analyzer = DivergenceAnalyzer()
    mem_analyzer = MemoryAnalyzer()

    filtered = [k for k in kernels if not kernel_filter or kernel_filter in k.name]
    if not filtered:
        rprint(f"[yellow]No kernels matching '{kernel_filter}'[/]")
        raise typer.Exit(0)

    if fmt == "json":
        results = []
        for kernel in filtered:
            occ_result = occ_analyzer.analyze(kernel, arch_spec, threads)
            occ_curve  = occ_analyzer.occupancy_curve(kernel, arch_spec) if curve else None
            div_result = div_analyzer.analyze(kernel)
            mem_result = mem_analyzer.analyze(kernel)
            results.append(build_json_report(kernel, occ_result, div_result, mem_result, occ_curve))
        payload = json.dumps(results, indent=2)
        if output:
            Path(output).write_text(payload, encoding="utf-8")
            rprint(f"[green]Results written to {output}[/]")
        else:
            print(payload)
        return

    rprint(f"\n[bold]cuda-sage[/] [dim]v{__version__}[/] — analyzing [cyan]{ptx_file}[/] "
           f"against [bold]{arch_spec.name}[/] ([dim]{arch_spec.sm}[/])\n")

    for kernel in filtered:
        occ_result  = occ_analyzer.analyze(kernel, arch_spec, threads)
        occ_curve   = occ_analyzer.occupancy_curve(kernel, arch_spec) if curve else None
        div_result  = div_analyzer.analyze(kernel)
        mem_result  = mem_analyzer.analyze(kernel)
        render_kernel_report(kernel, occ_result, div_result, mem_result, occ_curve)


# ─────────────────────────────────────────────────────────────────────────────
# Command:        diff
# Purpose:        Compare two PTX files by kernel name and report deltas.
# Inputs:         baseline, optimized — paths to PTX files
#                 arch — SM target for occupancy model
#                 threads — threads/block for occupancy calc
# Outputs:        Rich table with Δ occupancy, Δ regs, Δ spills, Δ divergence,
#                 and a IMPROVED / REGRESSION / NEUTRAL verdict per kernel.
# Preconditions:  Both files exist; at least one kernel name appears in both.
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def diff(
    baseline: Path = typer.Argument(..., help="Baseline PTX file"),
    optimized: Path = typer.Argument(..., help="Optimized PTX file to compare against baseline"),
    arch: str = typer.Option("sm_80", "--arch", "-a"),
    threads: int = typer.Option(256, "--threads", "-t"),
) -> None:
    """Compare two PTX files and report performance regressions or improvements."""
    for f in (baseline, optimized):
        if not f.exists():
            rprint(f"[red]Error:[/] File not found: {f}")
            raise typer.Exit(1)

    parser = PTXParser()
    arch_spec = get_arch(arch)
    occ_analyzer = OccupancyAnalyzer()
    div_analyzer = DivergenceAnalyzer()
    mem_analyzer = MemoryAnalyzer()

    base_kernels = {k.name: k for k in parser.parse_file(baseline)}
    opt_kernels  = {k.name: k for k in parser.parse_file(optimized)}

    common = set(base_kernels) & set(opt_kernels)
    if not common:
        rprint("[yellow]No matching kernel names found between the two files.[/]")
        raise typer.Exit(0)

    rprint(f"\n[bold]cuda-sage diff[/] — [cyan]{baseline.name}[/] → [cyan]{optimized.name}[/]\n")

    table = Table(title=f"Performance Delta ({arch_spec.sm})", box=box.ROUNDED)
    table.add_column("Kernel", style="bold")
    table.add_column("Occupancy Δ", justify="right")
    table.add_column("Regs Δ", justify="right")
    table.add_column("Spills Δ", justify="right")
    table.add_column("Divergence sites Δ", justify="right")
    table.add_column("Verdict")

    for name in sorted(common):
        bk, ok = base_kernels[name], opt_kernels[name]
        b_occ = occ_analyzer.analyze(bk, arch_spec, threads)
        o_occ = occ_analyzer.analyze(ok, arch_spec, threads)
        b_div = div_analyzer.analyze(bk)
        o_div = div_analyzer.analyze(ok)
        b_mem = mem_analyzer.analyze(bk)
        o_mem = mem_analyzer.analyze(ok)

        d_occ   = o_occ.occupancy - b_occ.occupancy
        d_regs  = ok.registers.physical_regs - bk.registers.physical_regs
        d_spill = sum(w.count for w in o_mem.spill_warnings) - sum(w.count for w in b_mem.spill_warnings)
        d_div   = len(o_div.sites) - len(b_div.sites)

        occ_str   = f"[{'green' if d_occ >= 0 else 'red'}]{d_occ:+.1%}[/]"
        regs_str  = f"[{'red' if d_regs > 0 else 'green' if d_regs < 0 else 'dim'}]{d_regs:+d}[/]"
        spill_str = f"[{'red' if d_spill > 0 else 'green' if d_spill < 0 else 'dim'}]{d_spill:+d}[/]"
        div_str   = f"[{'red' if d_div > 0 else 'green' if d_div < 0 else 'dim'}]{d_div:+d}[/]"

        is_regression = (d_occ < -0.05) or (d_regs > 8) or (d_spill > 0) or (d_div > 0)
        is_improvement = (d_occ > 0.05) or (d_regs < 0 and d_spill <= 0)
        verdict = "[red]REGRESSION[/]" if is_regression else ("[green]IMPROVED[/]" if is_improvement else "[dim]NEUTRAL[/]")

        table.add_row(name, occ_str, regs_str, spill_str, div_str, verdict)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Command:        list-archs
# Purpose:        Print a table of all supported SM architectures and their
#                 hardware resource limits.
# Side Effects:   Writes to stdout.
# ─────────────────────────────────────────────────────────────────────────────
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
            sm, arch.name,
            str(arch.max_warps_per_sm),
            str(arch.max_threads_per_sm),
            str(arch.regs_per_sm),
            f"{arch.smem_per_sm_bytes // 1024} KB",
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Command:        transform
# Purpose:        Apply source-to-source performance transforms to a CUDA C file.
# Inputs:         cu_file — path to .cu source
#                 output — where to write optimized source (default: <name>.optimized.cu)
#                 block_size — max threads/block for __launch_bounds__ annotation
#                 show_diff — if True, print a unified diff instead of writing file
# Side Effects:   Writes transformed source to output file (or stdout with --diff).
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def transform(
    cu_file: Path = typer.Argument(..., help="Path to .cu CUDA C source file to optimize"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path (default: <name>.optimized.cu)"),
    block_size: int = typer.Option(256, "--block-size", "-b", help="Max threads/block for __launch_bounds__"),
    show_diff: bool = typer.Option(False, "--diff", "-d", help="Print unified diff instead of writing file"),
) -> None:
    """Apply proven CUDA C performance transforms: __launch_bounds__, __restrict__,
    shared-mem bank-conflict padding, #pragma unroll, and warp-divergence hints."""
    from .transform import CUDASourceTransformer
    import difflib

    if not cu_file.exists():
        rprint(f"[red]Error:[/] File not found: {cu_file}")
        raise typer.Exit(1)

    source = cu_file.read_text()
    result = CUDASourceTransformer(default_block_size=block_size).transform(source)

    if not result.has_changes:
        rprint("[green]✓ No transforms needed — source is already optimized.[/]")
        raise typer.Exit(0)

    rprint(f"\n[bold]cuda-sage transform[/] — [cyan]{cu_file.name}[/]\n")

    # Summary table
    table = Table(title="Applied Transforms", box=box.SIMPLE_HEAD)
    table.add_column("Transform", style="bold")
    table.add_column("Line", justify="right")
    table.add_column("Impact", justify="center")
    table.add_column("Description")

    impact_colors = {"high": "red", "medium": "yellow", "low": "green"}
    for t in result.transforms:
        color = impact_colors.get(t.impact, "white")
        table.add_row(
            t.name, str(t.line_no),
            f"[{color}]{t.impact.upper()}[/{color}]",
            t.description[:80] + ("…" if len(t.description) > 80 else ""),
        )
    console.print(table)
    rprint(f"\n[bold]{len(result.transforms)}[/] transform(s) applied "
           f"([red]{result.high_impact_count} HIGH[/] impact).\n")

    if show_diff:
        diff = difflib.unified_diff(
            source.splitlines(keepends=True),
            result.transformed_source.splitlines(keepends=True),
            fromfile=str(cu_file),
            tofile=str(output or cu_file.with_suffix(".optimized.cu")),
        )
        sys.stdout.writelines(diff)
    else:
        out_path = output or cu_file.with_suffix(".optimized.cu")
        out_path.write_text(result.transformed_source)
        rprint(f"[green]✓ Optimized source written to[/] [cyan]{out_path}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Command:        tune
# Purpose:        Auto-tune kernel launch parameters using NVRTC + CUDA events,
#                 or the occupancy cost model when no GPU is available.
# Inputs:         cu_file — CUDA C source with #define tunable parameters
#                 kernel — entry point name
#                 arch — SM target string
#                 block_size_values — comma-separated block sizes to try
#                 strategy — "grid" or "random"
#                 n — problem size for GPU benchmarking
#                 no_gpu — force static model even if GPU is available
# Outputs:        Rich table showing all evaluated configurations + best result.
# ─────────────────────────────────────────────────────────────────────────────
@app.command()
def tune(
    cu_file: Path = typer.Argument(..., help="CUDA C source file with tunable #define parameters"),
    kernel: str = typer.Option("", "--kernel", "-k", help="Kernel entry point name"),
    arch: str = typer.Option("sm_80", "--arch", "-a", help="Target SM architecture"),
    block_sizes: str = typer.Option("", "--block-sizes", help="Comma-separated block sizes to try (e.g. 64,128,256,512)"),
    strategy: str = typer.Option("grid", "--strategy", "-s", help="Search strategy: grid or random"),
    n: int = typer.Option(1_000_000, "--n", help="Problem size for GPU benchmarking"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Force static model (no GPU benchmarking)"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use persistent result cache"),
) -> None:
    """Auto-tune kernel launch parameters. Uses NVRTC+CUDA events on NVIDIA GPUs,
    falls back to occupancy cost model otherwise. Results are cached in SQLite."""
    from .tune import KernelAutoTuner, SearchSpace, TuneParam, TuneCache

    if not cu_file.exists():
        rprint(f"[red]Error:[/] File not found: {cu_file}")
        raise typer.Exit(1)

    source = cu_file.read_text()
    kernel_name = kernel or cu_file.stem

    # Build search space
    if block_sizes:
        sizes = [int(x.strip()) for x in block_sizes.split(",")]
        space = SearchSpace([TuneParam("BLOCK_SIZE", sizes)], strategy=strategy)
    else:
        space = SearchSpace.from_source(source)
        space.strategy = strategy
        if not space.params:
            # Default: sweep standard block sizes
            space = SearchSpace([TuneParam("BLOCK_SIZE", [32, 64, 128, 256, 512, 1024])])

    cache = TuneCache() if use_cache else None

    rprint(f"\n[bold]cuda-sage tune[/] — [cyan]{cu_file.name}[/] ({kernel_name})\n")
    rprint(f"Search space: {space.size} configuration(s), strategy=[bold]{strategy}[/]\n")

    result = KernelAutoTuner().tune(
        source, kernel_name, space, arch=arch, n=n,
        cache=cache, force_model=no_gpu,
    )

    # Results table
    src_label = "[green]GPU[/]" if result.source == "gpu" else "[yellow]model[/]"
    table = Table(title=f"Tuning Results ({arch}, source: {src_label})", box=box.ROUNDED)
    for p in space.params:
        table.add_column(p.name, justify="right")
    table.add_column("Occupancy", justify="right")
    table.add_column("Time (ms-equiv)", justify="right")
    table.add_column("", justify="center")   # best marker

    for pt in sorted(result.all_points, key=lambda p: p.time_ms):
        is_best = pt.params == result.best_params
        marker = "[bold green]★ BEST[/]" if is_best else ""
        row = [str(pt.params.get(p.name, "—")) for p in space.params]
        row += [f"{pt.occupancy:.1%}", f"{pt.time_ms:.1f}", marker]
        table.add_row(*row)

    console.print(table)

    speedup_pct = (result.speedup - 1.0) * 100
    if result.improved:
        rprint(f"\n[bold green]★ {speedup_pct:.1f}% predicted speedup[/] with "
               + ", ".join(f"[bold]{k}={v}[/]" for k, v in result.best_params.items()))
    else:
        rprint(f"\n[yellow]Default configuration is near-optimal ({speedup_pct:+.1f}%)[/]")

    if result.recommendations:
        rprint("\n[bold]Recommendations:[/]")
        for i, rec in enumerate(result.recommendations, 1):
            rprint(f"  {i}. {rec}")
    rprint()


if __name__ == "__main__":
    app()
