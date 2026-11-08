"""Command line: serve, eval, gate.

    uv run mediamix serve            # API and UI on http://127.0.0.1:8859 (fits at startup, about 10 seconds)
    uv run mediamix eval             # full study, writes reports/eval.json and reports/summary.md
    uv run mediamix eval --gate      # same, exit 1 unless the well-identified channels recover true ROI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def money(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def summary_markdown(rep: dict) -> str:
    lines = ["# Study summary", "", f"{rep['brand']}: {rep['weeks']} weeks ({rep['first_week']} to "
             f"{rep['last_week']}), fit on {rep['train_weeks']}, holdout {rep['holdout_weeks']}.", ""]
    fb, fc = rep["fit"]["base"], rep["fit"]["calibrated"]
    lines += [f"Holdout MAPE: base {fb['holdout_mape']:.2%}, calibrated {fc['holdout_mape']:.2%}.", ""]
    lt = rep["lift_test"]
    lines += [f"Lift test ({lt['channel']}, {lt['regions']} regions, weeks {lt['test_weeks'][0]} to "
              f"{lt['test_weeks'][1]}): iROAS {lt['iroas']:.2f} (90% CI {lt['ci90'][0]:.2f} to {lt['ci90'][1]:.2f}), "
              f"true {lt['true_iroas']:.2f}.", ""]
    lines += ["## Recovered vs true ROI (training weeks, 90% block bootstrap intervals)", "",
              "| Channel | True ROI | Model | 90% CI | Error | Calibrated | 90% CI | Error |",
              "|---|---:|---:|---|---:|---:|---|---:|"]
    for r in rep["roi"]:
        b, c = r["base"], r["calibrated"]
        lines.append(f"| {r['label']} | {r['true_roi']:.2f} | {b['roi']:.2f} | {b['ci90'][0]:.2f} to {b['ci90'][1]:.2f}"
                     f" | {b['error_pct']:+.0f}% | {c['roi']:.2f} | {c['ci90'][0]:.2f} to {c['ci90'][1]:.2f} | "
                     f"{c['error_pct']:+.0f}% |")
    p = rep["plan"]
    lines += ["", f"## Budget plan, {money(p['quarter_budget'])} for the quarter (calibrated model)", "",
              "| Channel | Current | Optimized | Optimized 90% range |", "|---|---:|---:|---|"]
    for ch in rep["channels"]:
        k = ch["channel"]
        lo, hi = p["allocation_ci90"][k]
        lines.append(f"| {ch['label']} | {money(p['current']['weekly_spend'][k] * 13)} | "
                     f"{money(p['optimized']['weekly_spend'][k] * 13)} | {money(lo)} to {money(hi)} |")
    lines += ["", f"Predicted media revenue lift {p['lift_pct']:+.1f}% (90% range {p['lift_pct_ci90'][0]:+.1f}% to "
              f"{p['lift_pct_ci90'][1]:+.1f}%), true lift {p['true_lift_pct']:+.1f}%. Uncalibrated model's plan: "
              f"predicted {rep['plan_base']['lift_pct']:+.1f}%, true {rep['plan_base']['true_lift_pct']:+.1f}%.", ""]
    g = rep["gate"]
    lines += [f"## Gate (tolerance {g['tolerance_pct']:.0f}%): {'PASS' if g['pass'] else 'FAIL'}", ""]
    for ch in g["checks"]:
        lines.append(f"- {ch['check']}: recovered {ch['recovered']:.2f}, true {ch['true']:.2f}, "
                     f"error {ch['error_pct']:+.1f}%: {'pass' if ch['pass'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mediamix")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8859")))
    e = sub.add_parser("eval")
    e.add_argument("--seed", type=int, default=3)
    e.add_argument("--boot", type=int, default=200)
    e.add_argument("--out", default="reports")
    e.add_argument("--gate", action="store_true", help="exit 1 unless well-identified channels recover true ROI")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("mediamix.app:create_app", factory=True, host=args.host, port=args.port, access_log=False)
        return 0

    from .study import Study, StudyConfig

    study = Study(StudyConfig(seed=args.seed, n_boot=args.boot, n_boot_opt=min(100, args.boot))).run()
    rep = clean(study.report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(rep, indent=1))
    (out / "summary.md").write_text(summary_markdown(rep))
    print(summary_markdown(rep))
    if args.gate and not rep["gate"]["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
