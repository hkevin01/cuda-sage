# cuda-sage

Static PTX analysis for CUDA kernels.

cuda-sage is focused on one job: catching performance risks directly from PTX text before runtime profiling. It does not require a GPU for analysis, and it is designed for fast feedback in local development and CI.

## Scope

The current project scope is intentionally narrow:

- Parse PTX kernel metadata.
- Estimate occupancy from architecture limits.
- Flag thread-driven warp divergence patterns.
- Detect memory-pattern risks such as spills and potential bank conflicts.
- Compare PTX revisions for directional regression analysis.

Everything outside this scope has been removed to keep the codebase maintainable and focused.

## Quick Start

```bash
git clone https://github.com/hkevin01/cuda-sage
cd cuda-sage
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Compile CUDA to PTX, then analyze:

```bash
nvcc -ptx -arch=sm_86 mykernel.cu -o mykernel.ptx
cuda-sage analyze mykernel.ptx --arch sm_86 --threads 256 --curve
```

## CLI Commands

| Command | Purpose |
| --- | --- |
| `analyze` | Run occupancy, divergence, and memory analysis on PTX kernels. |
| `diff` | Compare baseline and optimized PTX by kernel name. |
| `list-archs` | Show supported architecture models. |

### Analyze

```bash
cuda-sage analyze kernel.ptx --arch sm_80
cuda-sage analyze kernel.ptx --arch sm_80 --curve
cuda-sage analyze kernel.ptx --arch sm_80 --format json --output report.json
```

### Diff

```bash
cuda-sage diff baseline.ptx optimized.ptx --arch sm_80
```

### List Architectures

```bash
cuda-sage list-archs
```

## Supported Architectures

| SM | Example GPU | Shared Mem / SM |
| --- | --- | --- |
| `sm_70` | Volta V100 | 96 KB |
| `sm_75` | Turing T4 / RTX 2080 | 64 KB |
| `sm_80` | Ampere A100 | 164 KB |
| `sm_86` | Ampere RTX 3080/3090 | 100 KB |
| `sm_89` | Ada RTX 4090 class | 100 KB |
| `sm_90` | Hopper H100 | 228 KB |

## Python API

```python
from cudasage import PTXParser, OccupancyAnalyzer, DivergenceAnalyzer, MemoryAnalyzer
from cudasage import get_arch

kernels = PTXParser().parse_file("kernel.ptx")
arch = get_arch("sm_80")

occ = OccupancyAnalyzer().analyze(kernels[0], arch, threads_per_block=256)
div = DivergenceAnalyzer().analyze(kernels[0])
mem = MemoryAnalyzer().analyze(kernels[0])

print(occ.occupancy, occ.limiting_factor)
print(len(div.sites))
print(mem.spill_ops)
```

## Project Layout

```text
src/cudasage/
├── __init__.py
├── cli.py
├── reporter.py
├── analyzers/
│   ├── occupancy.py
│   ├── divergence.py
│   └── memory.py
├── models/
│   └── architectures.py
└── parsers/
    └── ptx_parser.py

tests/
├── fixtures/
│   ├── divergent_kernel.ptx
│   ├── matmul.ptx
│   ├── reduction.ptx
│   └── vecadd.ptx
├── test_cli.py
├── test_divergence.py
├── test_fixtures.py
├── test_memory.py
├── test_occupancy.py
├── test_parser.py
├── test_public_api.py
└── test_reporter.py
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

## Documentation

- [docs/api-reference.md](docs/api-reference.md)
- [docs/analysis-guide.md](docs/analysis-guide.md)
- [docs/architecture-specs.md](docs/architecture-specs.md)

## Limitations

- PTX static analysis only. No SASS or runtime profiling replacement.
- Divergence and bank-conflict checks are heuristic by design.
- Always validate important findings with runtime tools on target hardware.

## License

MIT. See [LICENSE](LICENSE).
