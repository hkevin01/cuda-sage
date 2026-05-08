"""Tests for the CUDA kernel auto-tuner (static model path — no GPU needed)."""
import pytest
import tempfile
from pathlib import Path
from cudasage.tune import TuneParam, SearchSpace, KernelAutoTuner, TuneResult, TuneCache
from cudasage.tune.benchmark import _static_score

VECADD_SRC = """\
#define BLOCK_SIZE 256

__global__ void vecadd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""


# ── TuneParam ────────────────────────────────────────────────────────────────

def test_tuneparam_default_is_first_value():
    p = TuneParam("BLOCK_SIZE", [32, 64, 128])
    assert p.default == 32


def test_tuneparam_explicit_default():
    p = TuneParam("BLOCK_SIZE", [32, 64, 128], default=64)
    assert p.default == 64


def test_tuneparam_invalid_default_raises():
    with pytest.raises(ValueError):
        TuneParam("BLOCK_SIZE", [32, 64], default=99)


def test_tuneparam_empty_values_raises():
    with pytest.raises(ValueError):
        TuneParam("BLOCK_SIZE", [])


# ── SearchSpace ───────────────────────────────────────────────────────────────

def test_search_space_size():
    space = SearchSpace([
        TuneParam("BLOCK_SIZE", [64, 128, 256]),
        TuneParam("TILE_SIZE",  [8, 16]),
    ])
    assert space.size == 6


def test_search_space_configs_count():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256])])
    assert len(space.configs()) == 3


def test_search_space_default_config():
    space = SearchSpace([
        TuneParam("BLOCK_SIZE", [64, 128, 256], default=128),
    ])
    assert space.default_config() == {"BLOCK_SIZE": 128}


def test_search_space_from_source_detects_block_size():
    space = SearchSpace.from_source(VECADD_SRC)
    names = [p.name for p in space.params]
    assert "BLOCK_SIZE" in names


def test_search_space_from_source_default_matches_define():
    space = SearchSpace.from_source(VECADD_SRC)
    block_param = next(p for p in space.params if p.name == "BLOCK_SIZE")
    assert block_param.default == 256


def test_search_space_empty_source():
    space = SearchSpace.from_source("__global__ void k() {}")
    assert space.size == 1   # empty space → 1 config


def test_random_strategy_is_deterministic_with_seed():
    params = [
        TuneParam("BLOCK_SIZE", [64, 128, 256, 512]),
        TuneParam("TILE_SIZE", [8, 16, 32]),
    ]
    s1 = SearchSpace(params=params, strategy="random", max_trials=4, random_seed=99)
    s2 = SearchSpace(params=params, strategy="random", max_trials=4, random_seed=99)
    assert s1.configs() == s2.configs()


# ── KernelAutoTuner (static model, no GPU) ───────────────────────────────────

def test_tuner_returns_tune_result():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [128, 256, 512])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert isinstance(result, TuneResult)


def test_tuner_result_source_is_model():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [128, 256])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert result.source == "model"


def test_tuner_evaluates_all_configs():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256, 512])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert len(result.all_points) == 4


def test_tuner_best_params_in_space():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert result.best_params["BLOCK_SIZE"] in [64, 128, 256]


def test_tuner_speedup_is_positive():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256, 512])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert result.speedup > 0


def test_tuner_has_recommendations():
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert len(result.recommendations) > 0


def test_tuner_best_time_le_default():
    """Best time should be ≤ default config time (we always find the minimum)."""
    space = SearchSpace([TuneParam("BLOCK_SIZE", [64, 128, 256, 512])])
    result = KernelAutoTuner().tune(
        VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True
    )
    assert result.best_time_ms <= result.default_time_ms + 1e-9


def test_static_score_penalizes_large_shared_memory():
    src_small = """
__global__ void k(float* x) {
    __shared__ float tile[32];
    int i = threadIdx.x;
    x[i] = tile[i & 31];
}
"""
    src_large = """
__global__ void k(float* x) {
    __shared__ float tile[256][256];
    int i = threadIdx.x;
    x[i] = tile[0][i & 255];
}
"""
    t_small, _ = _static_score(src_small, 256, "sm_80")
    t_large, _ = _static_score(src_large, 256, "sm_80")
    assert t_large > t_small


# ── TuneCache ────────────────────────────────────────────────────────────────

def test_cache_put_and_get():
    from cudasage.tune.benchmark import BenchmarkPoint
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    cache = TuneCache(db_path=db)
    params = {"BLOCK_SIZE": 256}
    pt = BenchmarkPoint(params=params, time_ms=1.23, occupancy=0.75, source="model")
    cache.put("src", "sm_80", params, pt)
    got = cache.get("src", "sm_80", params)
    assert got is not None
    assert got.time_ms == pytest.approx(1.23)
    assert got.occupancy == pytest.approx(0.75)
    db.unlink()


def test_cache_miss_returns_none():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    cache = TuneCache(db_path=db)
    assert cache.get("no_such_source", "sm_80", {"BLOCK_SIZE": 256}) is None
    db.unlink()


def test_cache_stats():
    from cudasage.tune.benchmark import BenchmarkPoint
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    cache = TuneCache(db_path=db)
    pt = BenchmarkPoint(params={"BLOCK_SIZE": 256}, time_ms=1.0, occupancy=0.5, source="model")
    cache.put("src", "sm_80", {"BLOCK_SIZE": 256}, pt)
    stats = cache.stats()
    assert stats["total"] == 1
    assert stats["model"] == 1
    db.unlink()


def test_cache_clear():
    from cudasage.tune.benchmark import BenchmarkPoint
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    cache = TuneCache(db_path=db)
    pt = BenchmarkPoint(params={"X": 1}, time_ms=1.0, occupancy=0.5, source="model")
    cache.put("src", "sm_80", {"X": 1}, pt)
    removed = cache.clear()
    assert removed == 1
    assert cache.stats()["total"] == 0
    db.unlink()


def test_tuner_with_cache():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    cache = TuneCache(db_path=db)
    space = SearchSpace([TuneParam("BLOCK_SIZE", [128, 256])])
    tuner = KernelAutoTuner()
    # First run — populates cache
    r1 = tuner.tune(VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True, cache=cache)
    # Second run — should all come from cache
    r2 = tuner.tune(VECADD_SRC, "vecadd", space, arch="sm_80", force_model=True, cache=cache)
    assert r1.best_time_ms == pytest.approx(r2.best_time_ms)
    assert cache.stats()["total"] == 2   # 2 unique configs stored
    db.unlink()
