# Contributing to cuda-sage

Thank you for your interest in contributing! This document covers the development workflow, coding standards, and how to add new analyses or architecture support.

---

## Getting Started

```bash
git clone https://github.com/hkevin01/cuda-sage
cd cuda-sage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v   # should pass 98 tests
```

---

## Development Workflow

1. **Fork** the repository and create a branch: `git checkout -b feature/my-change`
2. **Make changes** incrementally with tests.
3. **Run the full suite** before pushing: `pytest -v`
4. **Open a pull request** against `main` using the PR template.

---

## Code Standards

### Style
- Python ≥ 3.11, type annotations throughout.
- No external runtime dependencies beyond `typer` and `rich`.
- Formatters: `ruff format` (line length 100); linter: `ruff check`.

### Specification comments
Every public function and class should carry structured block comments with these fields where relevant:

```python
# ─────────────────────────────────────────────────────────────────────────────
# Function:   my_function
# Purpose:    One sentence description.
# Inputs:     arg: type — description, valid range
# Outputs:    return type — description
# Failure Modes: what can go wrong and how it is handled
# ─────────────────────────────────────────────────────────────────────────────
def my_function(arg: int) -> str:
    ...
```

### Tests
- Every new feature must have corresponding tests in `tests/`.
- Use inline PTX strings for parser/analyzer unit tests rather than new fixture files where possible.
- Tests must be deterministic and fast (the full suite runs in < 1 second).

---

## Adding a New Architecture

1. Add a new entry to `ARCHITECTURES` in [src/cudasage/models/architectures.py](src/cudasage/models/architectures.py).
2. Verify `get_arch` fallback still works correctly.
3. Add a test asserting the new SM target exists in `ARCHITECTURES`.

All hardware specs must be sourced from the CUDA C Programming Guide (Appendix — Compute Capabilities).

---

## Adding a New Analysis Pass

1. Create `src/cudasage/analyzers/my_analysis.py` following the pattern of existing analyzers.
2. Define a result dataclass and an analyzer class with an `analyze(kernel: KernelInfo) -> MyResult` method.
3. Export the new symbols from `src/cudasage/__init__.py`.
4. Wire it into the CLI in `cli.py` and the reporter in `reporter.py` (both text and JSON paths).
5. Add tests in `tests/test_my_analysis.py`.

---

## Reporting Bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template. A minimal PTX reproducer is the fastest path to a fix.

---

## License

By contributing, you agree that your contributions will be licensed under the project's MIT license.
