# Changelog

All notable changes to cuda-sage are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] — 2026-04-21

### Added
- `--format json` option on `analyze` command — machine-readable output of all analysis results
- `--output <file>` option to write results to disk instead of stdout
- `build_json_report()` function in `reporter.py` for programmatic JSON serialization
- Full public API via `cudasage/__init__.py` — all 18 symbols exported from a single import
- 63 new tests (CLI integration, reporter/JSON, public API edge cases) — total: 98 tests
- Structured specification comments on all public functions and classes
- GitHub Actions CI workflow (test matrix Python 3.11–3.13, lint, CLI smoke tests)
- Standard GitHub project files: `.gitignore`, issue templates, PR template, CONTRIBUTING, SECURITY, CHANGELOG

### Changed
- Version bumped to `0.2.0`

---

## [0.1.0] — 2026-04-20

### Added
- PTX parser: extracts kernel metadata (registers, shared memory, instructions) from `.ptx` files
- `OccupancyAnalyzer`: full CUDA occupancy formula with 4 limiting factors and curve support
- `DivergenceAnalyzer`: forward taint propagation detecting thread-ID-driven branches
- `MemoryAnalyzer`: spill detection, bank conflict heuristic, missing sync warning, arithmetic intensity proxy
- `reporter.py`: Rich terminal output with panels, bar charts, color-coded severity
- CLI: `analyze`, `diff`, `list-archs` commands via Typer
- Architecture support: sm_70 (Volta) through sm_90 (Hopper)
- Test fixtures: `vecadd.ptx` (clean) and `divergent_kernel.ptx` (spills + divergence)
- 35 initial tests
