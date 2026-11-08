import numpy as np
import pytest

from mediamix.__main__ import clean, main, summary_markdown
from mediamix.study import QUARTER_WEEKS
from mediamix.world import CHANNELS


def test_gate_passes_for_well_identified_channels(study):
    g = study.report["gate"]
    assert g["pass"], g
    assert {c["check"] for c in g["checks"]} == {"Influencer", "Email", "Paid social + display combined"}


def test_roi_table_shape(study):
    rows = study.report["roi"]
    assert [r["channel"] for r in rows] == list(CHANNELS)
    for r in rows:
        for k in ("base", "calibrated"):
            lo, hi = r[k]["ci90"]
            assert lo <= hi


def test_known_failure_modes_show_up(study):
    rows = {r["channel"]: r for r in study.report["roi"]}
    assert not rows["podcast"]["base"]["recovered"] or rows["podcast"]["base"]["ci90"][0] == 0  # no variation
    assert not rows["display"]["base"]["recovered"]  # correlated with social, split goes wrong


def test_plan_respects_default_bounds(study):
    p = study.plan
    for c in CHANNELS:
        lo, hi = p["bounds"][c]
        assert lo - 1e-6 <= p["optimized"]["weekly_spend"][c] <= hi + 1e-6
    assert sum(p["optimized"]["weekly_spend"].values()) * QUARTER_WEEKS == pytest.approx(p["quarter_budget"], rel=1e-4)
    assert p["lift_pct"] > 0 and 0 <= p["prob_beats_current"] <= 1


def test_custom_budget_and_bounds(study):
    budget = 3_000_000.0
    w = budget / QUARTER_WEEKS
    lo = np.zeros(len(CHANNELS))
    hi = np.full(len(CHANNELS), w)
    hi[CHANNELS.index("tv")] = 0.0
    out = study.optimize_budget(budget, lo, hi, uncertainty=False)
    assert out["optimized"]["weekly_spend"]["tv"] == pytest.approx(0, abs=1e-6)
    assert "allocation_ci90" not in out


def test_scenarios(study):
    s = study.scenarios({"tv": 0.0})
    assert set(s) == {"current", "optimized", "user"}
    assert s["user"]["weekly_spend"]["tv"] == 0
    assert s["user"]["quarter_media_revenue"] < s["current"]["quarter_media_revenue"]
    lo, hi = s["current"]["quarter_media_revenue_ci90"]
    assert lo <= hi
    with pytest.raises(ValueError):
        study.scenarios({"radio": 1.0})


def test_response_curves_payload(study):
    k = study.response_curves()
    assert set(k) == set(CHANNELS)
    for v in k.values():
        assert len(v["spend"]) == len(v["revenue"]) == len(v["true_revenue"]) == len(v["revenue_lo"])
        assert all(a <= b + 1e-6 for a, b in zip(v["revenue_lo"], v["revenue_hi"], strict=True))


def test_decomposition_payload_sums(study):
    d = study.decomposition()
    total = np.sum([d["series"][k] for k in d["series"]], axis=0)
    assert np.allclose(total, d["fitted"], atol=0.1)
    assert d["holdout_start"] == 130


def test_summary_markdown_mentions_every_channel(study):
    md = summary_markdown(clean(study.report))
    for label in ("TV", "Paid search", "Podcast", "Email"):
        assert label in md
    assert "PASS" in md


def test_cli_eval_writes_reports(tmp_path):
    rc = main(["eval", "--boot", "10", "--out", str(tmp_path), "--gate"])
    assert rc == 0
    assert (tmp_path / "eval.json").exists() and (tmp_path / "summary.md").exists()
