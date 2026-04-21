# Analysis Guide

This guide explains the three static analysis passes and how to interpret their output.

---

## 1. Occupancy Analysis

**What it measures:** How many warps the SM can keep in flight simultaneously, as a fraction of the hardware maximum.

**Formula:**
```
occupancy = active_warps_per_sm / max_warps_per_sm

active_warps = active_blocks × warps_per_block
active_blocks = min(
    floor(max_threads_per_sm / threads_per_block),   # thread limit
    floor(regs_per_sm / (regs_per_warp)),            # register limit
    floor(smem_per_sm / smem_per_block),             # shared mem limit
    max_blocks_per_sm,                               # hardware limit
)
```

**Interpreting the output:**

| Occupancy | Interpretation |
|-----------|----------------|
| > 75% | Excellent — GPU will effectively hide memory latency |
| 50–75% | Good for compute-bound kernels |
| 25–50% | Marginal — latency hiding may be insufficient |
| < 25% | Poor — expect significant stalls |

**Limiting factors and fixes:**

| Factor | Cause | Typical Fix |
|--------|-------|-------------|
| `registers` | Too many registers per thread | `__launch_bounds__`, reduce temporaries, `-maxrregcount` |
| `shared_memory` | Large shared tile | Smaller tile size, dynamic smem tuning |
| `threads_per_block` | Block size too small or too large | Try 128, 256, 512; use the occupancy curve |
| `hw_block_limit` | Architecture hard cap | Normal for very large blocks |

**Occupancy curve (`--curve`):** Sweep the occupancy formula across standard block sizes (32 → 1024). Use this to find the block size that maximizes occupancy before experimenting.

---

## 2. Warp Divergence Detection

**What it detects:** Conditional branches (`bra`) whose predicate is derived from per-thread data (`%tid.x/y/z`, `%laneid`, `%warpid`).

**Why it matters:** When threads in a 32-thread warp take different paths, the SM serializes both paths, executing one while masking the other. In the worst case (50/50 split), effective throughput is halved.

**Taint propagation algorithm:**

1. **Seed:** Any register loaded from `%tid.x`, `%tid.y`, `%tid.z`, `%laneid`, or `%warpid` is marked *tainted*.
2. **Propagate:** Any arithmetic/logical/conversion instruction whose inputs include a tainted register taints its output.
3. **Predicate taint:** `setp` instructions with a tainted operand produce a tainted predicate register.
4. **Flag:** `@%p bra` using a tainted predicate is recorded as a divergence site.

**Severity levels:**

| Severity | Condition | Expected Impact |
|----------|-----------|-----------------|
| `high` | `rem.` or `and.` on thread ID upstream (odd/even split) | ~50% warp efficiency |
| `medium` | Branch predicated on thread ID without clear 50/50 pattern | Variable, profile to confirm |
| `low` | Thread-ID-derived branch with low expected divergence | Minimal |

**Common patterns:**

```ptx
// HIGH: odd/even split — 50% serialization
rem.u32     %r6, %tid.x, 2
setp.eq.u32 %p1, %r6, 0
@%p1 bra    $even_path

// MEDIUM: per-thread conditional load
ld.global.f32  %f1, [addr]
setp.gt.f32    %p2, %f1, 0f3f800000   // threshold depends on data
@%p2 bra       $positive_path
```

**Fixes:**
- Replace short divergent branches with `selp` (predicated select).
- Restructure so threads in a warp process the same data type.
- For persistent divergence (different work per thread), consider separate kernel launches.

---

## 3. Memory Pattern Analysis

**What it checks:**

### Register Spills (`ld.local` / `st.local`)
When the compiler runs out of register slots, it spills values to *local memory* — which is backed by global DRAM (~600 cycle latency, vs ~30 cycles for L1). Even a few spills per thread add up across thousands of threads.

```
Severity thresholds:
  > 10 ops → critical
  1–10 ops → warning
```

Fix: Use `__launch_bounds__(threads, minblocks)` to cap register usage, or split the kernel.

### Shared Memory Bank Conflicts
NVIDIA shared memory has 32 banks of 4 bytes each. If multiple threads in a warp access the same bank simultaneously, accesses are serialized (32-way conflict = 32× slower).

Detection is stride-based: if a `ld.shared`/`st.shared` instruction uses a literal constant byte offset that is a multiple of 128 (= 32 banks × 4 bytes), it is flagged as high risk.

Fix: Pad shared memory arrays by 1 element:
```cuda
__shared__ float smem[ROWS][COLS + 1];  // breaks stride-32 conflict
```

### Missing `bar.sync`
If a kernel writes to shared memory but has no `bar.sync` / `membar` instruction, threads may read stale values written by other threads. cuda-sage warns when `shared_stores > 0` and `sync_barriers == 0`.

Fix: Add `__syncthreads()` after writes and before reads in any shared memory tiling pattern.

### Arithmetic Intensity Proxy
```
AI = arithmetic_ops / global_mem_ops
```
- AI < 2: likely **memory-bandwidth bound** — focus on data reuse, vectorized loads
- AI > 10: likely **compute bound** — focus on ILP, warp occupancy

---

## Reading the JSON Output

`--format json` returns a list of per-kernel objects. Each has four top-level sections:

```json
{
  "kernel": "my_kernel",
  "sm_target": "sm_80",
  "overview": { "registers": 37, "shared_mem_bytes": 0, "instruction_count": 22 },
  "occupancy": {
    "value": 0.75, "percent": 75.0,
    "active_blocks": 6, "active_warps": 48,
    "limiting_factor": "registers",
    "suggestions": ["Register pressure is the bottleneck..."]
  },
  "divergence": {
    "has_divergence": false, "site_count": 0, "high_severity_count": 0,
    "tainted_reg_count": 0, "sites": [], "suggestions": []
  },
  "memory": {
    "global_loads": 2, "global_stores": 1,
    "spill_ops": 0, "sync_barriers": 0,
    "arithmetic_intensity_proxy": 5.0,
    "memory_bound_likely": false,
    "spill_warnings": [], "bank_conflict_risks": [], "suggestions": []
  }
}
```

Use this in CI to gate on regressions:

```python
import json, subprocess, sys

result = subprocess.run(
    ["cuda-sage", "analyze", "kernel.ptx", "--arch", "sm_80", "--format", "json"],
    capture_output=True, text=True,
)
data = json.loads(result.stdout)
for k in data:
    if k["occupancy"]["value"] < 0.5:
        print(f"FAIL: {k['kernel']} occupancy {k['occupancy']['percent']:.0f}% < 50%")
        sys.exit(1)
    if k["divergence"]["high_severity_count"] > 0:
        print(f"FAIL: {k['kernel']} has high-severity divergence")
        sys.exit(1)
```
