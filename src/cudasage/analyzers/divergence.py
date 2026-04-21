"""Warp divergence detector.

Warp divergence occurs when threads within a 32-thread warp take different
execution paths through a conditional branch. Only the worst case is:
  - Branch predicate derived directly from thread ID (threadIdx.x/y/z or lane ID)
  - Causes 50% or more of warps to serialize

Detection strategy:
  1. Find all "setp" instructions that compare against a value derived from
     %tid.x/y/z — these produce predicate registers used for conditional branches.
  2. Track which predicate registers are "tainted" by thread-ID arithmetic.
  3. Flag "@%p bra" instructions using tainted predicates.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from ..parsers.ptx_parser import KernelInfo


@dataclass
class DivergenceSite:
    line_no: int
    line_text: str
    predicate_reg: str
    severity: str   # "high" | "medium" | "low"
    reason: str


@dataclass
class DivergenceResult:
    kernel_name: str
    sites: list[DivergenceSite] = field(default_factory=list)
    tainted_regs: set[str] = field(default_factory=set)
    suggestions: list[str] = field(default_factory=list)

    @property
    def has_divergence(self) -> bool:
        return bool(self.sites)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for s in self.sites if s.severity == "high")


# Patterns that taint a destination register with thread-ID dependency
_RE_TID_SRC   = re.compile(r"%tid\.[xyz]|%laneid|%warpid")
_RE_SETP      = re.compile(r"setp\.(\S+)\s+(%p\w+),\s*(.*)")
_RE_COND_BRA  = re.compile(r"@(!?)(%p\w+)\s+bra")
_RE_ARITH_TID = re.compile(r"(add|sub|mul|mad|shl|shr|and|or|rem|div).*%tid\.[xyz]")
_RE_MOV_TID   = re.compile(r"mov\.\w+\s+(%r\w+),\s*%tid\.[xyz]")
_RE_DEST_REG  = re.compile(r"(%[rpfd]\w+),")  # first operand = destination

# Operations that propagate taint to a new register
_RE_TAINT_PROP = re.compile(
    r"(add|sub|mul|mad|shl|shr|and|or|rem|div|cvt|mov|neg|abs|selp|setp)\.\S+\s+(%\w+),"
)


class DivergenceAnalyzer:
    """Detect warp-divergent branches in a parsed PTX kernel.

    Algorithm:      Forward taint propagation over the instruction stream.
                    Seeds are PTX special registers known to differ per thread
                    (%tid.x/y/z, %laneid, %warpid). Any arithmetic instruction
                    whose source touches a tainted value propagates taint to
                    its destination. A conditional branch (@%p bra) on a
                    tainted predicate is classified as a divergence site.

    Limitations:    Intra-kernel, single-pass analysis. No loop unrolling or
                    alias analysis; false-positive rate is low but non-zero.
                    Branches on %ctaid.x (shared across warp) are not flagged.
    Thread Safety:  Stateless; safe for concurrent use.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Method:         analyze
    # Purpose:        Run taint propagation and collect divergence sites.
    # Inputs:         kernel: KernelInfo — fully parsed PTX kernel
    # Outputs:        DivergenceResult with sites, tainted_regs, suggestions
    # Preconditions:  kernel.instructions populated by PTXParser
    # Postconditions: result.tainted_regs ⊇ seeds from %tid.x/y/z/%laneid
    # Side Effects:   None (pure function).
    # ─────────────────────────────────────────────────────────────────────────
    def analyze(self, kernel: KernelInfo) -> DivergenceResult:
        result = DivergenceResult(kernel_name=kernel.name)
        tainted: set[str] = set()  # register names tainted by thread-ID values

        for instr in kernel.instructions:
            line = instr.operands
            op = instr.opcode
            full = f"{op} {line}"

            # ── Seed: any register loaded from %tid.x/y/z ─────────────
            if _RE_TID_SRC.search(full):
                m = _RE_DEST_REG.search(line)
                if m:
                    tainted.add(m.group(1))
                    result.tainted_regs.add(m.group(1))

            # ── Propagate: arithmetic on tainted registers ─────────────
            if any(t in full for t in tainted):
                pm = _RE_TAINT_PROP.match(full)
                if pm:
                    new_tainted = pm.group(2)
                    tainted.add(new_tainted)
                    result.tainted_regs.add(new_tainted)

            # ── setp on tainted value → tainted predicate ─────────────
            sm = _RE_SETP.match(f"{op} {line}")
            if sm:
                pred_reg = sm.group(2)
                rest = sm.group(3)
                # If either operand of setp is tainted, predicate is tainted
                if any(t in rest for t in tainted) or _RE_TID_SRC.search(rest):
                    tainted.add(pred_reg)
                    result.tainted_regs.add(pred_reg)

            # ── Conditional branch on tainted predicate ───────────────
            # The predicate is stored separately in instr.predicate (e.g. "%p2")
            if op in ("bra", "brx", "call") and instr.predicate:
                pred_reg = instr.predicate.lstrip("!")
                if pred_reg in tainted:
                    # Determine severity
                    # High: modulo/bitwise-AND with small constant (e.g. tid%2, tid&1)
                    # suggests 50%+ divergence
                    high_patterns = ["rem", "and", "mod"]
                    severity = "medium"
                    reason = f"Branch on predicate {pred_reg} derived from thread ID"

                    # Look back through instructions: if any rem/and op touched tainted regs,
                    # it means an odd/even split was computed from thread ID → high severity
                    for prev in kernel.instructions:
                        if ("rem." in prev.opcode or "and." in prev.opcode):
                            if any(t in prev.operands for t in tainted):
                                severity = "high"
                                reason = (
                                    f"Predicate {pred_reg} set by modulo/bitwise-and on thread ID "
                                    f"— likely 50% warp divergence (odd/even split pattern)"
                                )
                                break

                    src_line = (
                        kernel.source_lines[instr.line_no - 1]
                        if instr.line_no - 1 < len(kernel.source_lines)
                        else f"{op} {line}"
                    )
                    result.sites.append(DivergenceSite(
                        line_no=instr.line_no,
                        line_text=src_line.strip(),
                        predicate_reg=pred_reg,
                        severity=severity,
                        reason=reason,
                    ))

        result.suggestions = self._suggest(result)
        return result

    @staticmethod
    def _suggest(result: DivergenceResult) -> list[str]:
        suggestions = []
        if not result.has_divergence:
            return suggestions

        if result.high_severity_count > 0:
            suggestions.append(
                f"{result.high_severity_count} high-severity divergence site(s) detected. "
                "Odd/even thread splits serialize the entire warp — consider restructuring "
                "so all threads take the same path (predication over branching)."
            )

        suggestions.append(
            "Consider replacing divergent if/else with PTX predicated execution: "
            "use selp (select with predicate) for simple assignments, or "
            "ensure all threads in a warp follow the same branch when possible."
        )

        if len(result.sites) > 3:
            suggestions.append(
                f"{len(result.sites)} divergence sites found. Profile with "
                "Nsight Compute's 'Branch Efficiency' metric to prioritize which to fix first."
            )

        return suggestions
