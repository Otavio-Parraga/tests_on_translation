"""Method-awareness of the SV plumbing and the cross-method metrics.

Covers the two things that break silently when a new discovery method is added:
resume/de-dup keys that ignore the method (so RepE would be mistaken for a
finished CAA run) and cross-method comparison on the raw coefficient axis (which
rewards CAA for its scale convention rather than its direction).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from acttrans.constants import METHOD, METHODS, UNIT_NORM_METHODS
from acttrans.evaluation.method_compare import (
    analyze,
    analyze_block,
    best_per_method,
    block_key,
    normalize_method,
    summarize,
)
from acttrans.utils.paths import parse_sv_path, sv_path

_ROOT = Path(__file__).resolve().parent.parent


def _load_top_level(name):
    """Import a top-level script (ab_sweep.py etc.) as a module."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the registry ─────────────────────────────────────────────────────────────

def test_method_registry():
    assert METHOD in METHODS
    assert set(METHODS) == {"CAA", "RepE", "GCAV"}
    # the unit-norm set must be a subset, and CAA must NOT be in it: CAA's norm
    # carries behavior information, which is what makes the dose axis necessary
    assert set(UNIT_NORM_METHODS) <= set(METHODS)
    assert "CAA" not in UNIT_NORM_METHODS


@pytest.mark.parametrize("method", METHODS)
def test_sv_path_roundtrips_for_every_method(method):
    p = sv_path("root", "meta-llama/Llama-3.2-3B-Instruct", method,
                "sycophancy", "residual", 12)
    _, parsed_method, behavior, module, layer = parse_sv_path(p)
    assert (parsed_method, behavior, module, layer) == (method, "sycophancy",
                                                        "residual", 12)


# ── load_sv shape normalization ──────────────────────────────────────────────

def test_load_sv_normalizes_both_saved_shapes(tmp_path, monkeypatch):
    """CAA/RepE save [1, D]; GCAV saves [D]. Both must load as [D]."""
    from acttrans.comparison import common

    monkeypatch.setattr(common, "ACTENG_ROOT", tmp_path)
    for method, shape in [("RepE", (1, 16)), ("GCAV", (16,))]:
        p = sv_path(tmp_path, "m/M", method, "refusal", "residual", 8)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(shape, dtype=torch.bfloat16), p)
        vec = common.load_sv("m/M", "refusal", 8, method=method)
        assert vec.shape == (16,)
        assert vec.dtype == torch.float32


def test_load_sv_is_method_scoped(tmp_path, monkeypatch):
    """Asking for a method that has no vector must not silently fall back."""
    from acttrans.comparison import common

    monkeypatch.setattr(common, "ACTENG_ROOT", tmp_path)
    p = sv_path(tmp_path, "m/M", "CAA", "refusal", "residual", 8)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(1, 4), p)
    assert common.load_sv("m/M", "refusal", 8, method="CAA").shape == (4,)
    with pytest.raises(FileNotFoundError):
        common.load_sv("m/M", "refusal", 8, method="RepE")


def test_available_native_layers_is_method_scoped(tmp_path, monkeypatch):
    from acttrans.comparison import common

    monkeypatch.setattr(common, "ACTENG_ROOT", tmp_path)
    for method, layers in [("CAA", [4, 8]), ("GCAV", [8])]:
        for lay in layers:
            p = sv_path(tmp_path, "m/M", method, "refusal", "residual", lay)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.ones(1, 4), p)
    assert common.available_native_layers("m/M", "refusal", method="CAA") == [4, 8]
    assert common.available_native_layers("m/M", "refusal", method="GCAV") == [8]
    assert common.available_native_layers("m/M", "refusal", method="RepE") == []


# ── resume keys ──────────────────────────────────────────────────────────────

def test_combo_key_separates_methods_and_backfills_caa():
    ab_sweep = _load_top_level("ab_sweep")
    base = {"scope": "target", "translator": "t", "norm_mode": "restore",
            "behavior": "refusal"}

    # a row with no method (written before methods existed) is CAA, so old
    # results files keep resuming correctly instead of re-running
    assert ab_sweep._combo_key(base) == ab_sweep._combo_key({**base, "method": "CAA"})
    # and an empty method behaves the same as absent
    assert ab_sweep._combo_key({**base, "method": ""}) == \
        ab_sweep._combo_key({**base, "method": "CAA"})
    # different methods are different blocks — otherwise RepE would be skipped
    # as "already done" by the CAA rows
    keys = {ab_sweep._combo_key({**base, "method": m}) for m in METHODS}
    assert len(keys) == len(METHODS)


def test_ab_report_dedup_is_method_aware(tmp_path):
    ab_report = _load_top_level("ab_report")
    row = {"scope": "target", "translator": "t", "norm_mode": "restore",
           "behavior": "refusal", "coefficient": 1.0, "avg_p_match": 0.4,
           "accuracy": 0.5}
    f = tmp_path / "r.jsonl"
    import json
    f.write_text("\n".join(json.dumps(r) for r in [
        row,                          # -> CAA
        {**row, "method": "CAA"},     # duplicate of the above, must collapse
        {**row, "method": "RepE"},    # distinct method, must survive
    ]))

    rows = ab_report.load_rows([str(f)])
    assert len(rows) == 2
    assert sorted(r["method"] for r in rows) == ["CAA", "RepE"]
    # filtering keeps only the asked-for method
    assert [r["method"] for r in ab_report.load_rows([str(f)], methods=["RepE"])] == ["RepE"]


# ── backward compatibility of the CLI defaults ───────────────────────────────

def test_every_entry_point_defaults_to_caa_only():
    """Adding methods must not change what an existing caller does.

    Several committed shell wrappers (run_layer_sweep.sh, run_new_models.sh,
    scripts/run_ab_sweep.sh, ...) invoke these scripts with no method flag. If a
    default ever widened to all of METHODS, those wrappers would silently triple
    their work and mix methods into tables that share one coefficient axis.
    """
    cases = [
        ("ab_sweep", "--methods", ["CAA"]),
        ("compare_translated_and_original", "--methods", ["CAA"]),
        ("ab_dashboard", "--method", "CAA"),
        ("ab_pivot_dashboard", "--method", "CAA"),
    ]
    for name, flag, expected in cases:
        mod = _load_top_level(name)
        parser = _find_parser(mod)
        default = parser.get_default(flag.lstrip("-").replace("-", "_"))
        assert default == expected, f"{name}{flag} default is {default!r}, want {expected!r}"

    # ab_report filters nothing by default (it reports whatever is in the file)
    # but must warn rather than silently merge; None means "all rows present".
    rep = _load_top_level("ab_report")
    assert _find_parser(rep).get_default("methods") is None


def _find_parser(mod):
    """Build the module's ArgumentParser without running main()."""
    import argparse
    import contextlib
    import io

    captured = {}
    real_init = argparse.ArgumentParser.__init__

    def spy(self, *a, **kw):
        real_init(self, *a, **kw)
        captured.setdefault("p", self)

    argparse.ArgumentParser.__init__ = spy
    try:
        with contextlib.suppress(SystemExit), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            fn = getattr(mod, "parse_args", None) or mod.main
            import sys
            argv = sys.argv
            sys.argv = [argv[0], "--help"]
            try:
                fn()
            finally:
                sys.argv = argv
    finally:
        argparse.ArgumentParser.__init__ = real_init
    assert "p" in captured, "no ArgumentParser constructed"
    return captured["p"]


# ── cross-method metrics ─────────────────────────────────────────────────────

def _curve(method, sv_norm, pairs, scope="target", behavior="refusal",
           translator="t", target_layer=12):
    """Build the per-coefficient rows of one block from {coeff: p_match}."""
    return [{"method": method, "scope": scope, "translator": translator,
             "norm_mode": "restore", "behavior": behavior, "sv_norm": sv_norm,
             "target_layer": target_layer, "source_layer": 8,
             "coefficient": c, "avg_p_match": p, "accuracy": p, "n": 50}
            for c, p in pairs.items()]


def test_normalize_method_and_block_key():
    assert normalize_method({}) == "CAA"
    assert normalize_method({"method": None}) == "CAA"
    assert normalize_method({"method": "RepE"}) == "RepE"
    a = block_key({"scope": "target", "behavior": "refusal"})
    b = block_key({"method": "RepE", "scope": "target", "behavior": "refusal"})
    assert a != b


def test_dose_axis_makes_equal_directions_comparable():
    """The core claim: two vectors that are the same direction but differ only in
    norm convention must land on the SAME dose, even though their peak
    coefficients differ by exactly that norm ratio."""
    # CAA: norm 4, peaks at coeff 2  -> dose 8
    caa = analyze_block(_curve("CAA", 4.0, {-2: 0.1, -1: 0.3, 0: 0.5, 1: 0.7, 2: 0.9}))
    # RepE: unit norm, same curve shape but stretched 4x in coefficient
    repe = analyze_block(_curve("RepE", 1.0, {-8: 0.1, -4: 0.3, 0: 0.5, 4: 0.7, 8: 0.9}))

    assert caa["coeff_at_peak"] == 2 and repe["coeff_at_peak"] == 8
    # raw coefficients disagree 4x; doses agree exactly
    assert caa["dose_at_peak"] == pytest.approx(8.0)
    assert repe["dose_at_peak"] == pytest.approx(8.0)
    assert caa["dP_peak"] == pytest.approx(repe["dP_peak"])


def test_peak_search_ignores_the_collapsed_tail():
    """Past the coherent window P(match) -> 0 on BOTH sides, so those doses give
    ~0 swing and must not win the peak — this is what removes the need for an
    arbitrary |coeff| cutoff."""
    blk = analyze_block(_curve("CAA", 1.0, {
        -1000: 0.001, -5: 0.10, -1: 0.30, 0: 0.5, 1: 0.70, 5: 0.90, 1000: 0.002,
    }))
    assert blk["coeff_at_peak"] == 5           # not 1000
    assert blk["dP_peak"] == pytest.approx(0.80)


def test_peak_requires_a_symmetric_pair():
    """A positive coefficient with no negative twin cannot form a swing."""
    blk = analyze_block(_curve("CAA", 1.0, {0: 0.5, 1: 0.7, 4: 0.95, -1: 0.3}))
    assert blk["coeff_at_peak"] == 1           # 4 has no -4 partner
    assert blk["dP_peak"] == pytest.approx(0.4)


def test_monotonic_to_peak_detects_a_reversed_vector():
    rising = analyze_block(_curve("CAA", 1.0, {-2: 0.1, -1: 0.3, 0: 0.5, 1: 0.7, 2: 0.9}))
    assert rising["monotonic_to_peak"] == pytest.approx(1.0)
    # a vector wired backwards still has a peak, but it is negative
    falling = analyze_block(_curve("CAA", 1.0, {-2: 0.9, -1: 0.7, 0: 0.5, 1: 0.3, 2: 0.1}))
    assert falling["dP_peak"] < 0


def test_missing_sv_norm_degrades_to_nan_dose_not_a_crash():
    rows = _curve("CAA", 0.0, {-1: 0.3, 0: 0.5, 1: 0.7})
    blk = analyze_block(rows)
    assert blk["dP_peak"] == pytest.approx(0.4)     # swing still readable
    assert blk["dose_at_peak"] != blk["dose_at_peak"]  # dose is nan


def test_summarize_retention_is_matched_per_method_and_behavior():
    rows = []
    # native reference: CAA swings 0.8, RepE swings 0.4
    rows += _curve("CAA", 1.0, {-1: 0.1, 0: 0.5, 1: 0.9}, scope="native",
                   translator="native_l12")
    rows += _curve("RepE", 1.0, {-1: 0.3, 0: 0.5, 1: 0.7}, scope="native",
                   translator="native_l12")
    # translated: CAA recovers 0.4 of 0.8 = 50%; RepE recovers 0.4 of 0.4 = 100%
    rows += _curve("CAA", 1.0, {-1: 0.3, 0: 0.5, 1: 0.7}, scope="target")
    rows += _curve("RepE", 1.0, {-1: 0.3, 0: 0.5, 1: 0.7}, scope="target")

    summary = {(r["method"], r["scope"]): r for r in summarize(analyze(rows))}
    assert summary[("CAA", "target")]["mean_retention_vs_native"] == pytest.approx(0.5)
    assert summary[("RepE", "target")]["mean_retention_vs_native"] == pytest.approx(1.0)
    # a native row is its own reference
    assert summary[("CAA", "native")]["mean_retention_vs_native"] == pytest.approx(1.0)


def test_retention_skips_a_near_zero_native_reference():
    """A native vector that barely steers is not a usable denominator: dividing by
    it turns noise into retention of several hundred percent."""
    rows = []
    # native barely moves the behavior (swing 0.004, below the 0.02 floor)
    rows += _curve("CAA", 1.0, {-1: 0.498, 0: 0.5, 1: 0.502}, scope="native",
                   translator="native_l12")
    rows += _curve("CAA", 1.0, {-1: 0.3, 0: 0.5, 1: 0.7}, scope="target")

    summary = {(r["method"], r["scope"]): r for r in summarize(analyze(rows))}
    tgt = summary[("CAA", "target")]
    ret = tgt["mean_retention_vs_native"]
    assert ret != ret                      # nan, not 10000%
    assert tgt["n_retention_refs"] == 0
    assert tgt["n_no_native_ref"] == 1
    # lowering the floor lets the (meaningless) ratio through, proving the floor
    # is what suppressed it
    loose = {(r["method"], r["scope"]): r
             for r in summarize(analyze(rows), min_native_dP=0.0001)}
    assert loose[("CAA", "target")]["mean_retention_vs_native"] > 50


def test_best_per_method_picks_the_strongest_translator():
    rows = []
    rows += _curve("CAA", 1.0, {-1: 0.4, 0: 0.5, 1: 0.6}, translator="weak")
    rows += _curve("CAA", 1.0, {-1: 0.1, 0: 0.5, 1: 0.9}, translator="strong")
    best = best_per_method(analyze(rows), scope="target")
    assert len(best) == 1
    assert best[0]["translator"] == "strong"
    assert best[0]["dP_peak"] == pytest.approx(0.8)
