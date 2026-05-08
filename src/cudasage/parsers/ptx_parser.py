"""PTX assembly parser.

Extracts per-kernel metadata from NVIDIA PTX files without requiring
a GPU or CUDA toolkit — pure Python, zero native dependencies.

PTX reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RegisterFile:
    """Counts of each register type declared in a kernel."""
    pred: int = 0    # .pred — 1-bit predicate, ~1/8 physical reg each
    b16:  int = 0    # .b16  — half physical reg
    b32:  int = 0    # .b32  — 1 physical reg
    b64:  int = 0    # .b64  — 2 physical regs
    f16:  int = 0
    f32:  int = 0    # 1 physical reg
    f64:  int = 0    # 2 physical regs

    @property
    def physical_regs(self) -> int:
        """Estimate of 32-bit physical registers consumed per thread.

        Predicate regs share the 32-bit register file (1 pred ≈ 1/8 reg,
        rounded up in units of 8 per warp by the compiler).

        Rationale:   Hardware allocates the register file in 32-bit slots.
                     64-bit values occupy two adjacent slots. Predicate regs
                     are packed 8-per-slot but the compiler rounds warp
                     allocation up to the next multiple of 8.
        Outputs:     int ≥ 0 — estimated 32-bit register slots per thread.
        Failure Modes: Returns 0 when no registers are declared (valid for
                     devicelib helper functions).
        """
        pred_contrib = max(1, (self.pred + 7) // 8) if self.pred else 0
        return (
            pred_contrib
            + self.b16
            + self.b32 + self.f32
            + (self.b64 + self.f64) * 2
        )


@dataclass
class SharedMemoryDecl:
    name: str
    size_bytes: int
    align: int = 4


@dataclass
class Instruction:
    """A single decoded PTX instruction."""
    opcode: str       # e.g. "ld.global.f32"
    operands: str     # raw operand string
    line_no: int      # 1-based line number in the full PTX file
    kernel_line_no: int = 0  # 1-based line number relative to this kernel body
    source_line: str = ""    # exact source line text for precise reporting
    predicate: str = ""  # e.g. "%p2" or "!%p1" if guarded by @pred


@dataclass
class KernelInfo:
    """All statically-extractable information about one PTX kernel entry."""
    name: str
    sm_target: str                        # e.g. "sm_80"
    registers: RegisterFile = field(default_factory=RegisterFile)
    shared_mem_bytes: int = 0
    shared_decls: list[SharedMemoryDecl] = field(default_factory=list)
    instructions: list[Instruction] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)
    param_count: int = 0

    # ── Derived instruction counts (populated by parser) ──────────────────
    global_loads:   int = 0
    global_stores:  int = 0
    shared_loads:   int = 0
    shared_stores:  int = 0
    local_loads:    int = 0   # register spills — performance warning
    local_stores:   int = 0
    branches:       int = 0
    sync_barriers:  int = 0
    special_fns:    int = 0   # sin/cos/sqrt/rsqrt via mufu
    arithmetic:     int = 0

    @property
    def total_memory_ops(self) -> int:
        return self.global_loads + self.global_stores + self.shared_loads + self.shared_stores

    @property
    def arithmetic_intensity_proxy(self) -> float:
        """Ratio of arithmetic to global memory ops — higher = more compute-bound."""
        if self.global_loads + self.global_stores == 0:
            return float("inf")
        return self.arithmetic / (self.global_loads + self.global_stores)

    @property
    def spill_ops(self) -> int:
        return self.local_loads + self.local_stores


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Regex patterns for PTX constructs
_RE_VERSION    = re.compile(r"\.version\s+([\d.]+)")
_RE_TARGET     = re.compile(r"\.target\s+(sm_\d+)")
_RE_ENTRY      = re.compile(r"\.visible\s+\.entry\s+(\w+)|\.entry\s+(\w+)")
_RE_PARAM      = re.compile(r"\.param\s+\.\w+\s+\w+")
_RE_REG        = re.compile(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>|\.reg\s+\.(\w+)\s+%\w+;")
_RE_SHARED     = re.compile(r"\.shared\s+(?:\.align\s+(\d+)\s+)?\.b8\s+(\w+)\[(\d+)\]")
_RE_INSTR      = re.compile(r"^\s*(?:@(!?)(%[\w]+)\s+)?(\w[\w.]*)\s+(.*?);", re.MULTILINE)
_RE_BLOCK_END  = re.compile(r"^}")

# Instruction category prefixes
_GLOBAL_LD     = re.compile(r"^ld\.global")
_GLOBAL_ST     = re.compile(r"^st\.global")
_SHARED_LD     = re.compile(r"^ld\.shared")
_SHARED_ST     = re.compile(r"^st\.shared")
_LOCAL_LD      = re.compile(r"^ld\.local")
_LOCAL_ST      = re.compile(r"^st\.local")
_BRANCH        = re.compile(r"^bra|^brx|^call|^ret")
_BARRIER       = re.compile(r"^bar\.|^membar")
_SPECIAL_FN    = re.compile(r"^mufu|^sin\.|^cos\.|^sqrt\.|^rsqrt\.")
_ARITHMETIC    = re.compile(r"^(f?add|f?sub|f?mul|f?mad|fma|div|abs|neg|min|max|"
                             r"and|or|xor|shl|shr|mad|mul|cvt|mov|selp|set)")


class PTXParser:
    """Parse one or more PTX kernel entries from a .ptx file or string.

    Design:         Single-pass line scanner with regex-based pattern matching.
                    No external dependencies; suitable for CI and offline use.
    Thread Safety:  Stateless — safe to call parse_file / parse_string from
                    multiple threads simultaneously.
    Limitations:    Intra-kernel only; does not resolve .func call targets.
                    Instruction operand parsing is deliberately shallow (regex
                    over full PTX grammar) to remain fast and dependency-free.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         parse_file
    # Purpose:        Read a PTX file from disk and return all kernel entries.
    # Inputs:         path: str | Path — absolute or relative path to .ptx file
    # Outputs:        list[KernelInfo] — one entry per .entry directive found
    # Preconditions:  File must exist and be UTF-8 (or ASCII) encoded.
    # Failure Modes:  FileNotFoundError if path does not exist (propagated to
    #                 caller). UnicodeDecodeError suppressed via errors='replace'.
    # ─────────────────────────────────────────────────────────────────────────
    def parse_file(self, path: str | Path) -> list[KernelInfo]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.parse_string(text)

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         parse_string
    # Purpose:        Parse PTX source text and return all kernel entries.
    # Inputs:         text: str — complete PTX source
    # Outputs:        list[KernelInfo] — ordered list of parsed kernels
    # Postconditions: Each KernelInfo has instruction counts populated and
    #                 shared_mem_bytes reflects the sum of all .shared decls.
    # Side Effects:   None (pure function over input text).
    # ─────────────────────────────────────────────────────────────────────────
    def parse_string(self, text: str) -> list[KernelInfo]:
        lines = text.splitlines()
        kernels: list[KernelInfo] = []

        # Global SM target (applies to all kernels unless overridden)
        sm_target = "sm_80"
        m = _RE_TARGET.search(text)
        if m:
            sm_target = m.group(1)

        i = 0
        while i < len(lines):
            line = lines[i]
            em = _RE_ENTRY.search(line)
            if em:
                name = em.group(1) or em.group(2)
                kernel = KernelInfo(name=name, sm_target=sm_target)
                i = self._parse_kernel_body(lines, i + 1, kernel)
                kernels.append(kernel)
            else:
                i += 1

        return kernels

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         _parse_kernel_body
    # Purpose:        Parse the body of one .entry block, populating kernel.
    # Inputs:         lines: list[str] — all source lines
    #                 start: int — index of first line after the .entry header
    #                 kernel: KernelInfo — mutated in-place with all findings
    # Outputs:        int — index of the line after the closing brace
    # Preconditions:  lines[start:] must contain the kernel body and its '}'.
    # Postconditions: kernel.instructions, .registers, .shared_decls populated.
    # Side Effects:   Mutates kernel in place.
    # Failure Modes:  Malformed PTX (missing closing brace) returns len(lines).
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_kernel_body(self, lines: list[str], start: int, kernel: KernelInfo) -> int:
        """Parse from after the .entry line until the closing brace. Returns next line index."""
        depth = 0
        i = start
        while i < len(lines):
            raw = lines[i]
            line = raw.strip()

            # Track brace depth to find kernel end
            depth += line.count("{") - line.count("}")
            if depth < 0:
                return i + 1

            kernel.source_lines.append(raw)

            # Count .param declarations
            if _RE_PARAM.search(line):
                kernel.param_count += 1

            # Register declarations
            rm = _RE_REG.search(line)
            if rm:
                rtype = rm.group(1) or rm.group(3)
                count = int(rm.group(2)) if rm.group(2) else 1
                self._add_regs(kernel.registers, rtype, count)

            # Shared memory declarations
            sm = _RE_SHARED.search(line)
            if sm:
                align_str, sname, size_str = sm.group(1), sm.group(2), sm.group(3)
                size = int(size_str)
                align = int(align_str) if align_str else 4
                kernel.shared_decls.append(SharedMemoryDecl(sname, size, align))
                kernel.shared_mem_bytes += size

            # Instructions
            im = _RE_INSTR.match(raw)
            if im:
                neg, pred_reg, op, operands = im.group(1), im.group(2), im.group(3), im.group(4)
                if not op:
                    i += 1
                    continue
                pred = ("!" if neg else "") + (pred_reg or "")
                instr = Instruction(
                    opcode=op,
                    operands=operands or "",
                    line_no=i + 1,
                    kernel_line_no=len(kernel.source_lines),
                    source_line=raw.rstrip("\n"),
                    predicate=pred,
                )
                kernel.instructions.append(instr)
                self._classify_instruction(kernel, op)

            i += 1

        return i

    @staticmethod
    def _add_regs(regs: RegisterFile, rtype: str, count: int) -> None:
        mapping = {
            "pred": "pred", "b16": "b16", "b32": "b32", "b64": "b64",
            "f16": "f16", "f32": "f32", "f64": "f64",
            "u8": "b32", "u16": "b16", "u32": "b32", "u64": "b64",
            "s8": "b32", "s16": "b16", "s32": "b32", "s64": "b64",
        }
        attr = mapping.get(rtype, "b32")
        setattr(regs, attr, getattr(regs, attr) + count)

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         _classify_instruction
    # Purpose:        Increment the appropriate instruction-count field on kernel.
    # Inputs:         kernel: KernelInfo — mutated in-place
    #                 op: str — the PTX opcode string (e.g. "ld.global.f32")
    # Side Effects:   Exactly one counter on kernel is incremented per call.
    # Failure Modes:  Unknown opcodes are silently ignored (not counted).
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_instruction(kernel: KernelInfo, op: str) -> None:
        if _GLOBAL_LD.match(op):
            kernel.global_loads += 1
        elif _GLOBAL_ST.match(op):
            kernel.global_stores += 1
        elif _SHARED_LD.match(op):
            kernel.shared_loads += 1
        elif _SHARED_ST.match(op):
            kernel.shared_stores += 1
        elif _LOCAL_LD.match(op):
            kernel.local_loads += 1
        elif _LOCAL_ST.match(op):
            kernel.local_stores += 1
        elif _BRANCH.match(op):
            kernel.branches += 1
        elif _BARRIER.match(op):
            kernel.sync_barriers += 1
        elif _SPECIAL_FN.match(op):
            kernel.special_fns += 1
        elif _ARITHMETIC.match(op):
            kernel.arithmetic += 1
