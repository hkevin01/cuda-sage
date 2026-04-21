---
name: Feature Request
about: Suggest a new analysis pass, architecture, output format, or CLI option
labels: enhancement
---

## Summary

A concise description of the feature you'd like.

## Motivation

Why is this useful? What problem does it solve or what workflow does it improve?

## Proposed behavior

Describe the expected behavior, including example CLI invocations or API usage:

```bash
cuda-sage analyze kernel.ptx --arch sm_90 --new-flag
```

```python
from cudasage import NewAnalyzer
result = NewAnalyzer().analyze(kernel)
```

## Alternatives considered

What other approaches did you consider and why did you reject them?

## Additional context

Any references (CUDA docs, research papers, Nsight metrics) that informed this request.
