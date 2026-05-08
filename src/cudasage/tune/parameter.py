"""Tunable parameter definitions and search space management.

A SearchSpace is a cartesian product of TuneParam values.
The tuner iterates over (or samples from) this space to find the
configuration that minimises runtime.

Design rationale: Keep parameter definitions separate from the benchmark
engine so different search strategies (grid, random, Bayesian) can be
plugged in without changing the parameter model.
"""
from __future__ import annotations
import itertools
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuneParam:
    """A single tunable dimension.

    name:   The C preprocessor macro name this maps to (e.g. 'BLOCK_SIZE').
    values: Ordered list of candidate values to try.
    default: Which value to use as the baseline (defaults to values[0]).
    """
    name: str
    values: list[Any]
    default: Any = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"TuneParam '{self.name}' must have at least one value")
        if self.default is None:
            self.default = self.values[0]
        if self.default not in self.values:
            raise ValueError(
                f"TuneParam '{self.name}' default {self.default!r} not in values {self.values}"
            )


@dataclass
class SearchSpace:
    """Cartesian product of TuneParam values.

    Attributes:
        params: List of TuneParam objects defining the space.
        strategy: 'grid' (full enumeration) or 'random' (sampled subset).
        max_trials: Maximum number of configurations to evaluate (for random).

    Methods:
        configs():       Iterate over all configurations as dicts.
        default_config(): Return the dict of all param defaults.
        size:            Total number of configurations in the space.
    """
    params: list[TuneParam] = field(default_factory=list)
    strategy: str = "grid"       # "grid" | "random"
    max_trials: int = 64
    random_seed: int = 1337

    @property
    def size(self) -> int:
        n = 1
        for p in self.params:
            n *= len(p.values)
        return n

    def configs(self) -> list[dict[str, Any]]:
        """Return configurations to evaluate according to strategy."""
        if not self.params:
            return [{}]
        names = [p.name for p in self.params]
        all_configs = [
            dict(zip(names, combo))
            for combo in itertools.product(*[p.values for p in self.params])
        ]
        if self.strategy == "random" and len(all_configs) > self.max_trials:
            import random
            rng = random.Random(self.random_seed)
            return rng.sample(all_configs, self.max_trials)
        return all_configs

    def default_config(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    @classmethod
    def from_source(cls, source: str) -> "SearchSpace":
        """Auto-detect tunable parameters from CUDA source.

        Looks for lines of the form:
            #define BLOCK_SIZE 256     → TuneParam("BLOCK_SIZE", [32,64,128,256,512])
            #define TILE_SIZE  16      → TuneParam("TILE_SIZE",  [8,16,32])

        Any #define whose name contains 'BLOCK', 'TILE', 'CHUNK', 'WARP',
        'UNROLL', or 'SIZE' is treated as a tunable parameter.
        """
        # Patterns that suggest a parameter worth tuning
        TUNE_KEYWORDS = re.compile(
            r"BLOCK|TILE|CHUNK|WARP_SIZE|UNROLL|THREAD|SIZE",
            re.IGNORECASE,
        )
        RE_DEFINE = re.compile(r"#define\s+(\w+)\s+(\d+)", re.MULTILINE)

        params: list[TuneParam] = []
        seen: set[str] = set()

        for m in RE_DEFINE.finditer(source):
            name = m.group(1)
            default_val = int(m.group(2))
            if name in seen or not TUNE_KEYWORDS.search(name):
                continue
            seen.add(name)

            # Generate sensible candidate values around the default
            if "BLOCK" in name.upper() or "THREAD" in name.upper():
                candidates = [32, 64, 128, 256, 512, 1024]
            elif "TILE" in name.upper() or "SIZE" in name.upper():
                candidates = [4, 8, 16, 32, 64]
            elif "UNROLL" in name.upper():
                candidates = [1, 2, 4, 8, 16]
            else:
                candidates = [default_val // 2, default_val, default_val * 2]

            # Ensure default is in the candidate list
            if default_val not in candidates:
                candidates.append(default_val)
            candidates = sorted(set(candidates))

            params.append(TuneParam(name=name, values=candidates, default=default_val))

        return cls(params=params)
