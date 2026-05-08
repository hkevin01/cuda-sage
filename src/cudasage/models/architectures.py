"""GPU architecture specifications for occupancy modeling.

Covers NVIDIA SM architectures from Volta (sm_70) through Hopper (sm_90).
All specs sourced from CUDA C Programming Guide and NVIDIA Whitepaper datasheets.
"""
from __future__ import annotations
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Class:          Architecture
# Purpose:        Immutable record of hardware resource limits for one SM.
# Rationale:      Frozen dataclass prevents accidental mutation; all occupancy
#                 arithmetic is purely functional over these constants.
# Inputs:         Populated from ARCHITECTURES dict at module load time.
# Constraints:    All integer fields must be positive; warp_size == 32 for all
#                 current NVIDIA architectures.
# References:     CUDA C Programming Guide §K (Compute Capabilities)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Architecture:
    """Hardware limits for a single SM (Streaming Multiprocessor).

    Fields map directly to the four occupancy-limiting resources described in
    the CUDA occupancy calculator: thread slots, register file, shared memory,
    and the hardware block-count cap.
    """
    name: str               # e.g. "Ampere A100"
    sm: str                 # e.g. "sm_80"
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int        # total register file size per SM
    max_regs_per_thread: int
    smem_per_sm_bytes: int  # max configurable shared memory per SM
    smem_alloc_granularity: int  # shared mem allocated in multiples of this (bytes)
    reg_alloc_granularity: int   # regs allocated per warp in multiples of this
    warp_size: int = 32


ARCHITECTURES: dict[str, Architecture] = {
    "sm_70": Architecture(
        name="Volta V100", sm="sm_70",
        max_warps_per_sm=64, max_threads_per_sm=2048, max_blocks_per_sm=32,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=96 * 1024, smem_alloc_granularity=256,
        reg_alloc_granularity=8,
    ),
    "sm_75": Architecture(
        name="Turing T4/RTX 2080", sm="sm_75",
        max_warps_per_sm=32, max_threads_per_sm=1024, max_blocks_per_sm=16,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=64 * 1024, smem_alloc_granularity=256,
        reg_alloc_granularity=8,
    ),
    "sm_80": Architecture(
        name="Ampere A100", sm="sm_80",
        max_warps_per_sm=64, max_threads_per_sm=2048, max_blocks_per_sm=32,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=164 * 1024, smem_alloc_granularity=128,
        reg_alloc_granularity=8,
    ),
    "sm_86": Architecture(
        name="Ampere RTX 3080/3090", sm="sm_86",
        max_warps_per_sm=48, max_threads_per_sm=1536, max_blocks_per_sm=16,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=100 * 1024, smem_alloc_granularity=128,
        reg_alloc_granularity=8,
    ),
    "sm_89": Architecture(
        name="Ada Lovelace RTX 4090", sm="sm_89",
        max_warps_per_sm=48, max_threads_per_sm=1536, max_blocks_per_sm=24,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=100 * 1024, smem_alloc_granularity=128,
        reg_alloc_granularity=8,
    ),
    "sm_90": Architecture(
        name="Hopper H100", sm="sm_90",
        max_warps_per_sm=64, max_threads_per_sm=2048, max_blocks_per_sm=32,
        regs_per_sm=65536, max_regs_per_thread=255,
        smem_per_sm_bytes=228 * 1024, smem_alloc_granularity=128,
        reg_alloc_granularity=8,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Function:       get_arch
# Purpose:        Resolve a PTX SM target string to an Architecture record.
# Inputs:         sm: str — SM target string, e.g. "sm_80", "sm_90"
# Outputs:        Architecture — the matching or nearest-lower architecture
# Postconditions: Always returns a valid Architecture; never raises.
# Failure Modes:  Unknown string with no numeric suffix → falls back to sm_80.
#                 Requested version higher than all known → returns highest known.
# ─────────────────────────────────────────────────────────────────────────────
def get_arch(sm: str) -> Architecture:
    """Return the Architecture for a given sm target string.

    Falls back to the closest known architecture if exact match not found.
    """
    if not isinstance(sm, str) or not sm.strip():
        return ARCHITECTURES["sm_80"]

    sm = sm.strip()
    if sm in ARCHITECTURES:
        return ARCHITECTURES[sm]
    # Fallback: find nearest SM version <= requested
    try:
        ver = int(sm.replace("sm_", ""))
        candidates = {int(k.replace("sm_", "")): v for k, v in ARCHITECTURES.items()}
        best = max(k for k in candidates if k <= ver)
        return candidates[best]
    except (ValueError, TypeError):
        return ARCHITECTURES["sm_80"]
