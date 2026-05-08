# API Reference

This reference covers the focused public API for PTX static analysis.

```python
from cudasage import (
    PTXParser, KernelInfo, RegisterFile, Instruction,
    OccupancyAnalyzer, OccupancyResult, OccupancyCurvePoint,
    DivergenceAnalyzer, DivergenceResult, DivergenceSite,
    MemoryAnalyzer, MemoryResult, SpillWarning, BankConflictRisk,
    Architecture, ARCHITECTURES, get_arch,
    __version__,
)
```

## PTX Parser

### PTXParser

```python
parser = PTXParser()
kernels = parser.parse_file("kernel.ptx")
kernels = parser.parse_string(ptx_text)
```

| Method | Signature | Description |
| --- | --- | --- |
| parse_file | `(path: str | Path) -> list[KernelInfo]` | Parse PTX from disk. |
| parse_string | `(text: str) -> list[KernelInfo]` | Parse PTX from an in-memory string. |

### KernelInfo

`KernelInfo` holds parsed kernel metadata and derived instruction counts.

| Field | Type | Notes |
| --- | --- | --- |
| name | `str` | `.entry` kernel name. |
| sm_target | `str` | PTX target architecture, for example `sm_80`. |
| registers | `RegisterFile` | Register declarations aggregated by type. |
| shared_mem_bytes | `int` | Total static shared memory per block. |
| instructions | `list[Instruction]` | Parsed PTX instruction stream. |
| param_count | `int` | Number of `.param` declarations. |
| global_loads/global_stores | `int` | Global memory op counts. |
| shared_loads/shared_stores | `int` | Shared memory op counts. |
| local_loads/local_stores | `int` | Local memory ops, often spill indicators. |
| branches | `int` | Branch instruction count. |
| sync_barriers | `int` | `bar`/`membar` count. |
| special_fns | `int` | SFU-like operation count. |
| arithmetic | `int` | Arithmetic op proxy count. |

Derived properties:

- `total_memory_ops`
- `arithmetic_intensity_proxy`
- `spill_ops`

### Instruction

| Field | Type | Notes |
| --- | --- | --- |
| opcode | `str` | PTX opcode, for example `ld.global.f32`. |
| operands | `str` | Raw operand text without trailing semicolon. |
| line_no | `int` | 1-based line number in full PTX file. |
| kernel_line_no | `int` | 1-based line number relative to kernel body. |
| source_line | `str` | Original source line text for reporting. |
| predicate | `str` | Predicate guard, for example `%p1` or `!%p2`. |

## Occupancy Analyzer

### OccupancyAnalyzer

```python
occ = OccupancyAnalyzer().analyze(kernel, arch, threads_per_block=256)
curve = OccupancyAnalyzer().occupancy_curve(kernel, arch)
```

### OccupancyResult

| Field | Type | Notes |
| --- | --- | --- |
| occupancy | `float` | Theoretical occupancy in `[0.0, 1.0]`. |
| active_blocks | `int` | Concurrent blocks per SM. |
| active_warps | `int` | Concurrent warps per SM. |
| limiting_factor | `str` | Main bottleneck resource. |
| suggestions | `list[str]` | Human-readable guidance. |

`OccupancyCurvePoint` stores `threads_per_block`, `occupancy`, and `limiting_factor`.

## Divergence Analyzer

### DivergenceAnalyzer

```python
div = DivergenceAnalyzer().analyze(kernel)
```

This pass uses forward taint propagation from thread-varying sources (`%tid`, `%laneid`, `%warpid`) to identify predicates likely to diverge within a warp.

### DivergenceResult

| Field | Type | Notes |
| --- | --- | --- |
| kernel_name | `str` | Kernel analyzed. |
| sites | `list[DivergenceSite]` | Divergence locations. |
| tainted_regs | `set[str]` | Registers tainted by thread-varying data. |
| suggestions | `list[str]` | Mitigation tips. |

Properties:

- `has_divergence`
- `high_severity_count`

### DivergenceSite

| Field | Type | Notes |
| --- | --- | --- |
| line_no | `int` | PTX source line number. |
| line_text | `str` | Source text at the site. |
| predicate_reg | `str` | Predicate register controlling branch. |
| severity | `str` | `high`, `medium`, or `low`. |
| reason | `str` | Explanation for classification. |

## Memory Analyzer

### MemoryAnalyzer

```python
mem = MemoryAnalyzer().analyze(kernel)
```

### MemoryResult

| Field | Type | Notes |
| --- | --- | --- |
| global_load_count/global_store_count | `int` | Global memory operations. |
| shared_load_count/shared_store_count | `int` | Shared memory operations. |
| spill_warnings | `list[SpillWarning]` | Local memory spill risks. |
| bank_conflict_risks | `list[BankConflictRisk]` | Heuristic shared-memory conflict risks. |
| possible_missing_sync | `bool` | Shared stores without observed barrier. |
| suggestions | `list[str]` | Actionable guidance. |

Properties:

- `has_spills`
- `memory_bound_likely`
- `arithmetic_intensity_proxy`

### SpillWarning

| Field | Type | Notes |
| --- | --- | --- |
| kind | `str` | `load` or `store`. |
| count | `int` | Number of occurrences. |
| severity | `str` | Warning severity bucket. |

### BankConflictRisk

| Field | Type | Notes |
| --- | --- | --- |
| access_pattern | `str` | Instruction snippet that triggered risk. |
| stride_hint | `int | None` | Parsed stride if available. |
| risk_level | `str` | `low`, `medium`, or `high`. |
| description | `str` | Human-readable explanation. |

## Architecture Model

### get_arch

```python
arch = get_arch("sm_80")
```

Returns an `Architecture` record. If the target is unknown, it falls back to the nearest lower known architecture and defaults to `sm_80` for invalid strings.

### ARCHITECTURES

`ARCHITECTURES` is a mapping of supported SM targets to immutable `Architecture` dataclass instances.
