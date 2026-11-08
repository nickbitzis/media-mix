"""Media mix API and UI.

The study (simulate, search, fit, calibrate, bootstrap, plan) runs once at startup and lives in memory; nothing
is written to disk. The optimizer and scenario endpoints reuse the fitted curves and bootstrap draws, so each call
takes well under a second.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .__main__ import clean
from .logs import get_logger, log_event
from .metrics import Metrics
from .study import QUARTER_WEEKS, Study, StudyConfig
from .world import CHANNELS

STATIC = Path(__file__).parent / "static"
ModelName = Literal["base", "calibrated"]


class Bound(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class OptimizeIn(BaseModel):
    quarter_budget: float = Field(gt=0, le=1e9)
    bounds: dict[str, Bound] | None = None  # quarterly dollars per channel; missing channels use the defaults
    model: ModelName = "calibrated"
    uncertainty: bool = True


class ScenarioIn(BaseModel):
    user: dict[str, float] | None = None  # quarterly dollars per channel; missing channels keep current spend
    model: ModelName = "calibrated"


def load_study() -> Study:
    boot = int(os.environ.get("MEDIAMIX_BOOT", "200"))
    seed = int(os.environ.get("MEDIAMIX_SEED", "3"))
    study = Study(StudyConfig(seed=seed, n_boot=boot, n_boot_opt=min(100, boot))).run()
    log_event(get_logger(), "study fitted", seed=seed, bootstrap=boot, **study.report["timings"])
    return study


def create_app(study: Study | None = None) -> FastAPI:
    study = study or load_study()
    metrics = Metrics()
    log = get_logger()
    rep = clean(study.report)
    for r in rep["roi"]:
        metrics.set_gauge(f"mediamix_roi_error_pct_{r['channel']}", f"Calibrated ROI error vs truth: {r['label']}.",
                          r["calibrated"]["error_pct"])
    metrics.set_gauge("mediamix_holdout_mape", "Holdout MAPE of the calibrated model.",
                      rep["fit"]["calibrated"]["holdout_mape"])
    metrics.set_gauge("mediamix_gate_pass", "1 if the ROI recovery gate passes.", int(rep["gate"]["pass"]))

    app = FastAPI(title="Media Mix", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        path = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(log, "request", request_id=rid, method=request.method, path=request.url.path,
                      status=response.status_code, ms=round(dt * 1000, 1))
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "weeks": rep["weeks"], "gate_pass": rep["gate"]["pass"]}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/fit")
    def fit():
        return rep

    @app.get("/api/decomposition")
    def decomposition(model: ModelName = "calibrated"):
        return clean(study.decomposition(model))

    @app.get("/api/curves")
    def curves(model: ModelName = "calibrated"):
        return clean(study.response_curves(model))

    def bounds_arrays(body: OptimizeIn) -> tuple[np.ndarray, np.ndarray]:
        weekly = body.quarter_budget / QUARTER_WEEKS
        cur = study.current * weekly / study.current.sum()
        lo, hi = 0.5 * cur, 2.0 * cur
        for c, b in (body.bounds or {}).items():
            if c not in CHANNELS:
                raise HTTPException(422, f"unknown channel {c}")
            if b.max < b.min:
                raise HTTPException(422, f"{c}: max is below min")
            j = CHANNELS.index(c)
            lo[j], hi[j] = b.min / QUARTER_WEEKS, b.max / QUARTER_WEEKS
        return lo, hi

    @app.post("/api/optimize")
    def optimize(body: OptimizeIn):
        lo, hi = bounds_arrays(body)
        try:
            out = study.optimize_budget(body.quarter_budget, lo, hi, body.model, body.uncertainty)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        metrics.optimization(out["solver"])
        return clean(out)

    @app.post("/api/scenarios")
    def scenarios(body: ScenarioIn):
        if body.user and any(v < 0 for v in body.user.values()):
            raise HTTPException(422, "spend must be >= 0")
        try:
            return clean(study.scenarios(body.user, body.model))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
