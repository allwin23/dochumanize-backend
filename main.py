"""
DocHumanize - FastAPI Backend
Main application entry point with REST + WebSocket endpoints.
"""

import uuid
import asyncio
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import redis.asyncio as aioredis

from celery_app import celery_app
from celery.result import AsyncResult
import tempfile, os, shutil

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocHumanize API",
    description="AI-powered DOCX humanization pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUTPUT_DIR = "/tmp/dochumanize_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── WebSocket Connection Manager ─────────────────────────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections keyed by job_id."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id] = ws

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def send(self, job_id: str, payload: dict):
        ws = self.active.get(job_id)
        if ws:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                self.disconnect(job_id)


manager = ConnectionManager()


# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.post("/upload", summary="Upload DOCX and start humanization job")
async def upload_document(
    file: UploadFile = File(...),
    gemini_key: str = Form(...),
    hf_token: str = Form(...),
    evasion_mode: bool = Form(False),
):
    """
    Accepts a .docx file + Gemini API key.
    Queues a Celery background job and returns a job_id for WebSocket tracking.
    The API key is passed directly to the task and never persisted.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are accepted.")

    # Save upload to a temp file the worker can access
    job_id = str(uuid.uuid4())
    tmp_input = os.path.join(tempfile.gettempdir(), f"{job_id}_input.docx")

    with open(tmp_input, "wb") as f:
        content = await file.read()
        f.write(content)

    # Dispatch Celery task — key lives only in task args (in-memory in broker)
    from tasks import humanize_document
    task = humanize_document.apply_async(
        args=[tmp_input, gemini_key, hf_token, job_id, evasion_mode],
        task_id=job_id,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}", summary="Poll job status (fallback for no-WS clients)")
async def job_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    return {
        "job_id": job_id,
        "state": result.state,
        "info": result.info if isinstance(result.info, dict) else {},
    }


@app.get("/download/{job_id}", summary="Download humanized DOCX")
async def download_result(job_id: str):
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}_humanized.docx")
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Result not ready or job_id invalid.")
    return FileResponse(
        path=out_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="humanized_report.docx",
    )


# ── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """
    Real-time progress stream for a humanization job.
    The Celery worker publishes progress events to a Redis channel.
    This WS handler subscribes and forwards them to the client.

    Message schema:
      { "event": "progress", "page": 3, "para": 12, "total_paras": 150, "pct": 8 }
      { "event": "done", "download_url": "/download/<job_id>" }
      { "event": "error", "detail": "..." }
    """
    await manager.connect(job_id, websocket)
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"progress:{job_id}")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                payload = json.loads(message["data"])
                await manager.send(job_id, payload)

                # Close cleanly when terminal events arrive
                if payload.get("event") in ("done", "error"):
                    break

    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"progress:{job_id}")
        await redis.aclose()
        manager.disconnect(job_id)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}