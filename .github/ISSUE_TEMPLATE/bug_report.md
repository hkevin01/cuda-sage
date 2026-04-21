---
name: Bug Report
about: Report a defect in analysis results, CLI behavior, or output
labels: bug
---

## Description

A clear, concise description of what the bug is.

## PTX snippet or file

Paste a minimal PTX kernel that reproduces the issue, or attach a `.ptx` file:

```ptx
.version 7.0
.target sm_80
.entry repro_kernel(...) {
    ...
}
```

## Command used

```bash
cuda-sage analyze kernel.ptx --arch sm_80 --threads 256
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include full output or error messages:

```
paste output here
```

## Environment

- OS: 
- Python version: 
- cuda-sage version (`cuda-sage --version`):
- Install method: `pip install cuda-sage` / `pip install -e .`
