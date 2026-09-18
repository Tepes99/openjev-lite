"""FastAPI server for testing openjev-lite decisions in the browser.

Serves the static page and three endpoints:
  GET  /api/examples  — the openjev example scenarios (decisions.jsonl)
  GET  /api/info      — model name and endpoint for the header
  POST /api/score     — a row object or a list of rows -> scored results

Usage:
    uv run python server.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

import openjev_lite

ROOT = Path(__file__).parent
app = FastAPI(title="openjev-lite tester")


@app.get("/api/info")
def info() -> dict:
    ep = openjev_lite.load_endpoint()
    return {"model": ep["model"], "base_url": ep["base_url"]}


@app.get("/api/examples")
def examples() -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / "decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]


@app.post("/api/score")
async def score(request: Request) -> dict:
    payload = await request.json()
    rows = payload if isinstance(payload, list) else [payload]
    try:
        endpoint = openjev_lite.load_endpoint()
        results = [openjev_lite.score_row(row, endpoint) for row in rows]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"results": results}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8377)
