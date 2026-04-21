# API Reference

`cuda-sage` exposes a clean Python library API. Import from the top-level package:

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

---

## Parser

### `PTXParser`

Stateless parser. Safe to reuse across multiple files.

```python
parser = PTXParser()
kernels: list[KernelInfo] = parser.parse_file("mykernel.ptx")
kernels: list[KernelInfo] = parser.parse_string(ptx_text)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `parse_file` | `(path: str \| Path) → list[KernelInfo]` | Parse a PTX file from disk |
| `parse_string` | `(text: str) → list[KernelInfo]` | Parse PTX source text directly |

---

### `KernelInfo`

Dataclass populated by `PTXParser`. All fields are read-only after parse.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Kernel entry name |
| `sm_target` | `str` | PTX `.target` directive, e.g. `"sm_80"` |
| `registers` | `RegisterFile` | Register declarations |
| `shared_mem_bytes` | `int` | Total `.shared` memory in bytes |
| `instructions` | `list[Instruction]` | All parsed instructions |
| `param_count` | `int` | Number of `.param` declarations |
| `global_loads` | `int` | Count of `ld.global.*` instructions |
| `global_stores` | `int` | Count of `st.global.*` instructions |
| `shared_loads` | `int` | Count of `ld.shared.*` instructions |
| `shared_stores` | `int` | Count of `st.shared.*` instructions |
| `local_loads` | `int` | Count of `ld.local.*` (register spills) |
| `local_stores` | `int` | Count of `st.local.*` (register spills) |
| `sync_barriers` | `int` | Count of `bar.sync` / `membar` instructions |
| `special_fns` | `int` | Count of SFU ops (`mufu.sin/cos/sqrt`) |
| `arithmetic` | `int` | Count of arithmetic instructions |

**Derived properties:**

| Property | Description |
|----------|-------------|
| `total_memory_ops` | `global_loads + global_stores + shared_loads + shared_stores` |
| `arithmetic_intensity_proxy` | `arithmetic / (global_loads + global_stores)` |
| `spill_ops` | `local_loads + local_stores` |

---

### `RegisterFile`

```python
@dataclass
class RegisterFile:
    pred: int   # .pred registers
    b16:  int   # .b16 registers
    b32:  int   # .b32 registers
    b64:  int   # .b64 registers (2 physical regs each)
    f16:  int
    f32:  int   # 1 physical reg
    f64:  int   # 2 physical regs

    @property
    def physical_regs(self) -> int: ...  # Estimated 32-bit slots per thread
```

---

## Occupancy Analyzer

### `OccupancyAnalyzer`

```python
analyzer = OccupancyAnalyzer()

# Single block size
result: OccupancyResult = analyzer.analyze(kernel, arch, threads_per_block=256)

# Sweep of block sizes
curve: list[OccupancyCurvePoint] = analyzer.occupancy_curve(
    kernel, arch,
    thread_counts=[32, 64, 128, 256, 512, 1024],  # optional
)
```

### `OccupancyResult`

| Field | Type | Description |
|-------|------|-------------|
| `occupancy` | `float` | Theoretical occupancy [0.0 – 1.0] |
| `active_blocks` | `int` | Concurrent blocks per SM |
| `active_warps` | `int` | Concurrent warps per SM |
| `threads_per_block` | `int` | Input launch config |
| `limiting_factor` | `str` | `"registers"`, `"shared_memory"`, `"threads_per_block"`, or `"hw_block_limit"` |
| `blocks_by_threads` | `int` | Block limit from thread count |
| `blocks_by_regs` | `int` | Block limit from register file |
| `blocks_by_smem` | `int` | Block limit from shared memory |
| `blocks_by_hw_limit` | `int` | Hardware block cap |
| `suggestions` | `list[str]` | Human-readable tuning advice |

---

## Divergence Analyzer

### `DivergenceAnalyzer`

```python
result: DivergenceResult = DivergenceAnalyzer().analyze(kernel)
```

**Algorithm:** Forward taint propagation. Seeds are `%tid.x/y/z`, `%laneid`, `%warpid`. Propagates through arithmetic. `setp` on tainted value → tainted predicate. `@%p bra` on tainted predicate → divergence site.

### `DivergenceResult`

| Field | Type | Description |
|-------|------|-------------|
| `kernel_name` | `str` | Source kernel name |
| `sites` | `list[DivergenceSite]` | All detected divergence sites |
| `tainted_regs` | `set[str]` | All registers tainted by thread ID |
| `suggestions` | `list[str]` | Tuning advice |
| `has_divergence` | `bool` (property) | True when `sites` is non-empty |
| `high_severity_count` | `int` (property) | Count of `severity == "high"` sites |

### `DivergenceSite`

| Field | Type | Description |
|-------|------|-------------|
| `line_no` | `int` | Source line number |
| `line_text` | `str` | Source line content |
| `predicate_reg` | `str` | The tainted predicate register |
| `severity` | `str` | `"high"` (rem/and patterns), `"medium"`, `"low"` |
| `reason` | `str` | Human-readable explanation |

---

## Memory Analyzer

### `MemoryAnalyzer`

```python
result: MemoryResult = MemoryAnalyzer().analyze(kernel)
```

### `MemoryResult`

| Field | Type | Description |
|-------|------|-------------|
| `global_load_count` | `int` | Global load count |
| `global_store_count` | `int` | Global store count |
| `shared_load_count` | `int` | Shared load count |
| `shared_store_count` | `int` | Shared store count |
| `spill_warnings` | `list[SpillWarning]` | Local memory spill events |
| `bank_conflict_risks` | `list[BankConflictRisk]` | Heuristic bank conflict detections |
| `sync_barriers` | `int` | Sync barrier count |
| `possible_missing_sync` | `bool` | True when smem writes without any barrier |
| `suggestions` | `list[str]` | Tuning advice |
| `has_spills` | `bool` (property) | True when spill_warnings non-empty |
| `memory_bound_likely` | `bool` (property) | Global mem ratio > 40% of total ops |
| `arithmetic_intensity_proxy` | `float` (property) | arithmetic / global_mem_ops |

---

## Architecture Model

### `get_arch(sm: str) → Architecture`

```python
from cudasage import get_arch, ARCHITECTURES

arch = get_arch("sm_80")   # Ampere A100
arch = get_arch("sm_999")  # falls back to nearest-lower (sm_90)
```

Falls back to the nearest-lower known architecture. Never raises.

### `ARCHITECTURES`

```python
ARCHITECTURES: dict[str, Architecture]
# Keys: "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"
```

### `Architecture` fields

| Field | Description |
|-------|-------------|
| `name` | Human name, e.g. `"Ampere A100"` |
| `sm` | Target string, e.g. `"sm_80"` |
| `max_warps_per_sm` | Maximum concurrent warps per SM |
| `max_threads_per_sm` | Maximum concurrent threads per SM |
| `max_blocks_per_sm` | Hardware block-count cap |
| `regs_per_sm` | Total 32-bit register file slots per SM |
| `max_regs_per_thread` | Per-thread register cap |
| `smem_per_sm_bytes` | Configurable shared memory per SM |
| `smem_alloc_granularity` | Shared mem allocation granularity (bytes) |
| `reg_alloc_granularity` | Register allocation granularity per warp |
| `warp_size` | Always 32 for all supported architectures |

---

## Source Transformer

### `CUDASourceTransformer`

Applies five performance transformation passes to CUDA C/C++ source code. Output compiles cleanly and produces identical numerical results.

```python
from cudasage.transform import CUDASourceTransformer, TransformResult, Transform

transformer = CUDASourceTransformer(
    default_block_size=256,   # used in __launch_bounds__(N)
    max_unroll_count=16,      # loops with bound > N are not unrolled
)
result: TransformResult = transformer.transform(source_code: str)
```

**Transform passes (applied in order):**

| Pass | Name | Description | Impact |
|------|------|-------------|--------|
| T1 | `__launch_bounds__` | Injects `__launch_bounds__(N)` on kernels lacking it | `high` |
| T2 | `__restrict__` | Adds `__restrict__` to pointer params | `medium` |
| T3 | `shared_mem_padding` | Pads inner dims that are multiples of 16 to +1 | `high` |
| T4 | `#pragma unroll` | Prepends `#pragma unroll N` on constant-bound for-loops | `medium` |
| T5 | `divergence_hint` | Inserts `__ballot_sync` rewrite comment on `threadIdx%N` branches | `high` |

### `TransformResult`

| Field / Property | Type | Description |
|------------------|------|-------------|
| `original_source` | `str` | Input source |
| `transformed_source` | `str` | Output after all passes |
| `transforms` | `list[Transform]` | Records of each applied transformation |
| `has_changes` | `bool` | True when output differs from input |
| `high_impact_count` | `int` | Count of `impact == "high"` transforms |
| `summary()` | `list[str]` | One-line description per applied transform |

### `Transform`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Transform identifier (e.g. `"__launch_bounds__"`) |
| `description` | `str` | Human-readable description |
| `line_no` | `int` | Source line where transform was applied |
| `original` | `str` | Original code fragment |
| `replacement` | `str` | Replacement code fragment |
| `category` | `str` | `"register"`, `"memory"`, `"compute"`, or `"divergence"` |
| `impact` | `str` | `"high"`, `"medium"`, or `"low"` |
