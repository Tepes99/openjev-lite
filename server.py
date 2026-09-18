"""FastAPI server for testing the decision systems in the browser.

Serves the static page and endpoints:
  GET  /            — the tester page
  GET  /health      — liveness probe (used by the ship health check)
  GET  /api/info    — served model name (the laya base method)
  GET  /api/examples — the openjev example scenarios (decisions.jsonl)
  GET  /api/laya   — local laya model status (loading / ready / error)
  POST /api/score   — one row -> every decider scored together

The laya model is a one-time, somewhat slow load, so it is pre-warmed in a
background thread at startup; the server is ready to serve (and the page
loads) well before laya finishes.

Usage:
    uv run python server.py
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

import openjev_lite
import ml_systems

ROOT = Path(__file__).parent
_laya_warm_thread = threading.Thread(target=ml_systems.prewarm_laya, daemon=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off the one-time laya load in the background; do not block startup.
    if not _laya_warm_thread.is_alive():
        _laya_warm_thread.start()
    yield


app = FastAPI(title="decision systems tester", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/info")
def info() -> dict:
    return {"model": "laya · convaiinnovations/laya"}


@app.get("/api/examples")
def examples() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]


@app.get("/api/laya")
def laya_status() -> dict:
    """Non-blocking laya status: never loads on this path."""
    if ml_systems._laya_agent is not None:
        return {"status": "ready"}
    if ml_systems._laya_load_error:
        return {"status": "error", "error": ml_systems._laya_load_error}
    return {"status": "loading"}


@app.post("/api/score")
async def score(request: Request) -> dict:
    payload = await request.json()
    row = payload[0] if isinstance(payload, list) else payload
    try:
        openjev_lite.validate_row(row)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"results": ml_systems.score_all(row)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8377)
