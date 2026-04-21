# cuda-sage

**CUDA/PTX Static Analysis & Guidance Engine** — GPU-free static analyzer for NVIDIA PTX assembly that surfaces warp divergence, register spills, and occupancy bottlenecks before you ever touch a profiler.

```
cuda-sage analyze kernel.ptx --arch sm_80 --curve
cuda-sage diff baseline.ptx optimized.ptx --arch sm_86
cuda-sage list-archs
```

---

## Why

NVIDIA's ecosystem gives you post-execution profilers (Nsight Compute, nvprof). Those tools are powerful, but they require a working GPU, a compiled binary, a runtime environment, and time to run.

`cuda-sage` takes a different angle: **analyze PTX assembly statically**. You get actionable occupancy, divergence, and memory feedback at compile time — in CI, on a laptop without a GPU, or during code review.

---

## Quick Start

```bash
git clone https://github.com/yourname/cuda-sage
cd cuda-sage
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Compile CUDA to PTX (requires nvcc):
nvcc -ptx -arch=sm_86 mykernel.cu -o mykernel.ptx

# Analyze:
cuda-sage analyze mykernel.ptx --arch sm_86 --threads 256 --curve
```

Requires Python ≥ 3.11. No GPU, no CUDA toolkit, no compiled binaries needed.

---

## Features

### Occupancy Analysis
Implements the full CUDA occupancy formula from the CUDA C Programming Guide:

- All 4 limiting factors: thread count, register file, shared memory, HW block limit
- Architecture-specific register and shared memory allocation granularities
- Occupancy curve across block sizes (via `--curve`)
- Actionable suggestions for the identified bottleneck

### Warp Divergence Detection
Forward taint-propagation analysis through PTX register flow:

- Taint seeds: `%tid.x/y/z`, `%laneid`, `%warpid` (per-thread special registers)
- Propagates through arithmetic, conversions, and moves
- Classifies `setp` with tainted operands → tainted predicate
- Detects guarded `@%p bra` → divergence site
- **High severity** for `rem.` / `and.` patterns (odd/even thread splits → 50% serialization)

### Memory Pattern Analysis
- Global, shared, and local (spill) load/store counts
- Register spill detection (`ld.local` / `st.local`) with latency warning (~600 cycles vs L1 ~30 cycles)
- Shared memory bank conflict risk from stride patterns
- Missing `bar.sync` after shared writes
- Arithmetic intensity proxy (ops-per-global-mem-op) with compute/memory-bound classification

### Regression Diff
Compare two PTX files (baseline vs optimized) for the same kernel set. Reports:
- Δ occupancy, Δ registers, Δ spills, Δ divergence sites
- Overall verdict: `IMPROVED` / `REGRESSION` / `NEUTRAL`

### JSON Output
Machine-readable output for CI integration and toolchain pipelines:

```bash
cuda-sage analyze kernel.ptx --arch sm_80 --format json
cuda-sage analyze kernel.ptx --arch sm_80 --format json --output report.json
```

---

## Architecture Support

| SM Target | GPU                     | Max Warps/SM | Shared Mem/SM |
|-----------|-------------------------|:------------:|:-------------:|
| sm_70     | Volta V100              | 64           | 96 KB         |
| sm_75     | Turing T4 / RTX 2080    | 32           | 64 KB         |
| sm_80     | Ampere A100             | 64           | 164 KB        |
| sm_86     | Ampere RTX 3080/3090    | 48           | 100 KB        |
| sm_89     | Ada Lovelace RTX 4090   | 48           | 100 KB        |
| sm_90     | Hopper H100             | 64           | 228 KB        |

---

## Usage

### Analyze a PTX file

```bash
cuda-sage analyze mykernel.ptx --arch sm_86 --threads 256 --curve
```

```
cuda-sage v0.2.0 — analyzing mykernel.ptx against Ampere RTX 3080/3090 (sm_86)

─────────────────── Kernel: divergent_kernel  (sm_86) ────────────────────────
╭──────────────── Overview ──────────────────────────────────────────────────╮
│ PTX target     sm_86                                                       │
│ Registers      61 physical (32 ×32b, 14 ×64b, 4 pred)                     │
│ Shared memory  none                                                        │
│ Instructions   26                                                          │
╰────────────────────────────────────────────────────────────────────────────╯
╭──────────────── Occupancy ─────────────────────────────────────────────────╮
│   66.7%  ████████████████████░░░░░░░░░░  [32/32 warps, 4 blocks]          │
╰────────────────────────────────────────────────────────────────────────────╯
  Limiting factor: registers

  ⚠ 2 HIGH-severity divergence site(s) detected
    Line 27  %p1  HIGH  Odd/even thread split (rem.u32 on tid.x)
    Line 38  %p2  HIGH  Odd/even thread split (rem.u32 on tid.x)
```

### Compare baseline vs optimized

```bash
cuda-sage diff baseline.ptx optimized.ptx --arch sm_80
```

```
Performance Delta (sm_80)
╭──────────────┬─────────────┬────────┬──────────┬────────────────┬──────────╮
│ Kernel       │ Occupancy Δ │ Regs Δ │ Spills Δ │ Divergence Δ   │ Verdict  │
├──────────────┼─────────────┼────────┼──────────┼────────────────┼──────────┤
│ matmul       │     +16.7%  │    -8  │       -4 │             -2 │ IMPROVED │
╰──────────────┴─────────────┴────────┴──────────┴────────────────┴──────────╯
```

### JSON output for CI

```bash
cuda-sage analyze mykernel.ptx --arch sm_80 --format json
```

```json
[
  {
    "kernel": "vecadd",
    "sm_target": "sm_80",
    "occupancy": {
      "value": 0.75,
      "percent": 75.0,
      "limiting_factor": "registers",
      "suggestions": ["Register pressure is the bottleneck (37 regs/thread)..."]
    },
    "divergence": { "has_divergence": false, "site_count": 0 },
    "memory": { "spill_ops": 0, "global_loads": 2, "global_stores": 1 }
  }
]
```

### List supported architectures

```bash
cuda-sage list-archs
```

---

## Python Library API

Each analyzer is independent and importable:

```python
from cudasage import PTXParser, OccupancyAnalyzer, DivergenceAnalyzer, MemoryAnalyzer
from cudasage import get_arch

kernels = PTXParser().parse_file("mykernel.ptx")
arch = get_arch("sm_80")

# Occupancy
result = OccupancyAnalyzer().analyze(kernels[0], arch, threads_per_block=256)
print(f"Occupancy: {result.occupancy:.1%} — limited by {result.limiting_factor}")

# Divergence
div = DivergenceAnalyzer().analyze(kernels[0])
print(f"Divergence sites: {len(div.sites)}, tainted regs: {len(div.tainted_regs)}")

# Memory
mem = MemoryAnalyzer().analyze(kernels[0])
print(f"Spill ops: {mem.spill_ops}, AI proxy: {mem.arithmetic_intensity_proxy:.1f}")

# Occupancy curve
curve = OccupancyAnalyzer().occupancy_curve(kernels[0], arch)
for pt in curve:
    print(f"  {pt.threads_per_block:4d} threads → {pt.occupancy:.1%}  ({pt.limiting_factor})")
```

---

## Design Principles

**No GPU required.** The analyzer parses PTX text with regex-based pattern matching and applies static taint analysis. The occupancy formula is pure arithmetic from NVIDIA's CUDA C Programming Guide.

**Composable.** Each analyzer (`OccupancyAnalyzer`, `DivergenceAnalyzer`, `MemoryAnalyzer`) is independent, stateless, and reentrant. Use them standalone or combine for a full analysis pipeline.

**CI-friendly.** Exit code is non-zero only on errors; warnings are informational. JSON output enables downstream tooling:

```bash
cuda-sage analyze *.ptx --arch sm_80 --format json --output report.json
# Parse report.json in your CI pipeline to gate on regressions
```

**Documented engineering.** Every significant function carries structured specification blocks — purpose, inputs, outputs, preconditions, postconditions, failure modes — derived from safety-critical software engineering standards.

---

## Project Structure

```
src/cudasage/
├── __init__.py          # Public API: all symbols exported here
├── models/
│   └── architectures.py # SM70→SM90 hardware specs (frozen dataclasses)
├── parsers/
│   └── ptx_parser.py    # PTX → KernelInfo: registers, shared mem, instructions
├── analyzers/
│   ├── occupancy.py     # CUDA occupancy formula, 4 limiting factors, curve
│   ├── divergence.py    # Forward taint propagation, setp detection, bra flagging
│   └── memory.py        # Spills, bank conflicts, missing sync, intensity proxy
├── reporter.py          # Rich terminal output + JSON serialization
└── cli.py               # Typer CLI: analyze, diff, list-archs

tests/
├── fixtures/
│   ├── vecadd.ptx            # Clean kernel (no divergence, coalesced access)
│   └── divergent_kernel.ptx  # Problem kernel (divergence, spills, SFU ops)
├── test_parser.py        # 11 parser tests
├── test_occupancy.py     # 10 occupancy tests
├── test_divergence.py    # 6 divergence tests
├── test_memory.py        # 8 memory tests
├── test_cli.py           # 24 CLI integration tests
├── test_reporter.py      # 13 reporter / JSON tests
└── test_public_api.py    # 26 public API + edge case tests
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest -v
# 98 passed in 0.28s
```

Coverage spans:
- PTX parser (single kernel, multiple kernels, shared memory, register types)
- Occupancy formula (all 4 limiting factors, curve, edge cases)
- Divergence taint propagation (seeds, propagation, severity classification)
- Memory pattern analysis (spills, bank conflicts, missing sync, intensity)
- CLI integration (all commands, JSON output, `--output` file, error paths)
- Reporter (JSON serialization, Rich rendering, curve serialization)
- Public API (all exports, edge cases, frozen dataclasses)

---

## Installation

```bash
pip install cuda-sage        # from PyPI (when published)
# or from source:
git clone https://github.com/yourname/cuda-sage
cd cuda-sage
pip install -e ".[dev]"
```

---

## Limitations & Roadmap

**Current limitations:**
- PTX only — not SASS/cubin. For register counts on real compiled binaries, `cuobjdump --dump-ptx` works well.
- Taint analysis is intra-kernel only; no inter-procedural analysis across `call` targets.
- Bank conflict detection is heuristic (stride-based) rather than exhaustive.

**Roadmap:**
- SASS parser for post-lowering (post-register-allocation) analysis
- Loop unroll detection and trip-count inference
- Per-kernel `--exit-code 1 on regression` mode for strict CI gates
- HTML report output

---

## License

MIT
