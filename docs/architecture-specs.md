# Architecture Specifications

cuda-sage includes hardware resource tables for all major NVIDIA SM architectures from Volta through Hopper.

## Supported Architectures

| SM Target | GPU Family | Example GPUs | Max Warps/SM | Max Threads/SM | Shared Mem/SM |
|-----------|------------|--------------|:------------:|:--------------:|:-------------:|
| `sm_70` | Volta | V100, Titan V | 64 | 2048 | 96 KB |
| `sm_75` | Turing | T4, RTX 2080, GTX 1660 | 32 | 1024 | 64 KB |
| `sm_80` | Ampere | A100, A30 | 64 | 2048 | 164 KB |
| `sm_86` | Ampere | RTX 3080, 3090, A10 | 48 | 1536 | 100 KB |
| `sm_89` | Ada Lovelace | RTX 4090, 4080, L40 | 48 | 1536 | 100 KB |
| `sm_90` | Hopper | H100, H200 | 64 | 2048 | 228 KB |

## Detailed Specifications

### sm_70 — Volta V100

| Property | Value |
|----------|-------|
| Max warps/SM | 64 |
| Max threads/SM | 2048 |
| Max blocks/SM | 32 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 96 KB |
| Smem alloc granularity | 256 bytes |
| Reg alloc granularity | 8 registers/warp |

### sm_75 — Turing T4/RTX 2080

| Property | Value |
|----------|-------|
| Max warps/SM | 32 |
| Max threads/SM | 1024 |
| Max blocks/SM | 16 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 64 KB |
| Smem alloc granularity | 256 bytes |
| Reg alloc granularity | 8 registers/warp |

**Note:** Turing cuts max warps/SM and max blocks/SM in half compared to Volta and Ampere. This makes occupancy more sensitive to register pressure.

### sm_80 — Ampere A100

| Property | Value |
|----------|-------|
| Max warps/SM | 64 |
| Max threads/SM | 2048 |
| Max blocks/SM | 32 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 164 KB |
| Smem alloc granularity | 128 bytes |
| Reg alloc granularity | 8 registers/warp |

**Note:** A100 has the largest shared memory per SM (up to 164 KB with carveout), enabling larger matrix tiles.

### sm_86 — Ampere RTX 3080/3090

| Property | Value |
|----------|-------|
| Max warps/SM | 48 |
| Max threads/SM | 1536 |
| Max blocks/SM | 16 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 100 KB |
| Smem alloc granularity | 128 bytes |
| Reg alloc granularity | 8 registers/warp |

**Note:** Consumer Ampere (sm_86) has fewer warps per SM than A100 (sm_80). A kernel with 64 warps/SM on A100 achieves only 75% occupancy on sm_86.

### sm_89 — Ada Lovelace RTX 4090

| Property | Value |
|----------|-------|
| Max warps/SM | 48 |
| Max threads/SM | 1536 |
| Max blocks/SM | 24 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 100 KB |
| Smem alloc granularity | 128 bytes |
| Reg alloc granularity | 8 registers/warp |

**Note:** Ada raises max blocks/SM from 16 to 24 vs consumer Ampere, enabling better occupancy for small-block kernels.

### sm_90 — Hopper H100

| Property | Value |
|----------|-------|
| Max warps/SM | 64 |
| Max threads/SM | 2048 |
| Max blocks/SM | 32 |
| Register file/SM | 65,536 × 32-bit |
| Max registers/thread | 255 |
| Shared memory/SM | 228 KB |
| Smem alloc granularity | 128 bytes |
| Reg alloc granularity | 8 registers/warp |

**Note:** Hopper H100 has 228 KB shared memory per SM — the largest of any supported architecture. Also introduces warp group cooperative instructions (not modeled by cuda-sage's static analysis).

---

## Architecture Fallback Behavior

When `--arch` is set to an unknown SM target, cuda-sage falls back to the nearest-lower known architecture:

```
sm_88  → sm_86
sm_91  → sm_90
sm_999 → sm_90
"xyz"  → sm_80  (default for non-numeric strings)
```

This means analysis is always conservative — hardware limits are those of the nearest-lower known device.

---

## Adding New Architectures

See the [Contributing Guide](../CONTRIBUTING.md#adding-a-new-architecture) for step-by-step instructions on adding support for new SM targets (e.g. sm_100 / Blackwell).

Source of truth: NVIDIA CUDA C Programming Guide Appendix K (Compute Capabilities), available at https://docs.nvidia.com/cuda/cuda-c-programming-guide/.
