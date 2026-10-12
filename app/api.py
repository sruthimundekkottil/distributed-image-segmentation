"""HTTP API.

Two ways to get a prediction:

* Asynchronous (the "production" path):
    POST /jobs        -> puts the image on a Redis queue, returns a job ID immediately
    GET  /jobs/{id}   -> poll until status is "finished", then read the result
  Separate worker processes do the actual inference.

* Batches (many images at once, used by the web page served at "/"):
    POST /batches              -> one job per image, returns a batch ID
    GET  /batches/{id}         -> progress: total / completed / failed + per-job status
    GET  /batches/{id}/export  -> one combined COCO JSON for the whole batch

* Synchronous baseline (for comparison):
    POST /predict_sync -> runs the model inside the API process and waits for it

Run locally with:  uvicorn app.api:app --port 8000
"""

import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app import coco, config, metrics
from app.models import get_model
from app.tasks import predict_image

log = logging.getLogger("api")

# The web page (plain HTML/CSS/JS, no build step) lives next to this file.
STATIC_DIR = Path(__file__).parent / "static"

# Keys of a worker's return value that are bookkeeping, not part of the model result.
JOB_EXTRA_KEYS = ("inference_ms", "output_files")


def validate_image(data: bytes, file_name: str = "Upload") -> None:
    """Fail fast with HTTP 400/413 instead of letting a worker choke on a bad upload."""
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{file_name} is larger than {config.MAX_UPLOAD_BYTES} bytes")
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(400, f"{file_name} is not a valid image") from exc


def job_error(job: Job) -> str:
    """Last line of a failed job's traceback, e.g. "ValueError: bad input"."""
    result = job.latest_result()
    return result.exc_string.strip().splitlines()[-1] if result else "failed"


def create_app(redis: Redis | None = None) -> FastAPI:
    """Build the FastAPI app. Tests pass in a fake Redis connection."""
    redis = redis or Redis.from_url(config.REDIS_URL)
    queue = Queue(config.QUEUE_NAME, connection=redis)

    # A dedicated Prometheus registry that only contains our Redis-backed metrics.
    registry = CollectorRegistry()
    registry.register(metrics.RedisMetricsCollector(redis, queue))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the model up front for /predict_sync so the first request isn't slow.
        # The async /jobs path does not need the model in the API at all.
        try:
            get_model()
        except FileNotFoundError as exc:
            log.warning("Model not loaded, /predict_sync will fail: %s", exc)
        yield

    app = FastAPI(title="Scalable inference service", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    def fetch_job(job_id: str) -> Job:
        try:
            return Job.fetch(job_id, connection=redis)
        except NoSuchJobError as exc:
            raise HTTPException(404, "Job not found (unknown ID or result expired)") from exc

    def enqueue_images(images: list[tuple[str, bytes]], batch_id: str | None = None) -> list[Job]:
        """Put one job per (file_name, image_bytes) on the queue, or reject them all.

        Backpressure: if the workers are already far behind, refuse new work now
        instead of accepting it and making every client wait longer and longer.
        A batch is all-or-nothing: if it doesn't fit in the queue, nothing is enqueued.
        (The check and the enqueue are two separate steps, so under heavy
        concurrency the queue can overshoot the limit slightly. That's fine here.)
        """
        depth = len(queue)
        if depth + len(images) > config.MAX_QUEUE_DEPTH:
            metrics.increment(redis, metrics.JOBS_REJECTED, len(images))
            hint = " or send a smaller batch." if len(images) > 1 else "."
            raise HTTPException(
                429,
                f"Not enough room in the queue: {depth} job(s) waiting + {len(images)} new "
                f"would exceed the limit of {config.MAX_QUEUE_DEPTH}. Try again later{hint}",
                headers={"Retry-After": "1"},
            )

        # We enqueue the function by its import path, so the API never needs to
        # import worker-only code. time.time() lets the worker measure queue wait.
        # enqueue_many sends all jobs to Redis in a single round trip.
        jobs = [
            Queue.prepare_data(
                "app.tasks.run_inference",
                args=(data, time.time()),
                result_ttl=config.RESULT_TTL_SECONDS,
                failure_ttl=config.RESULT_TTL_SECONDS,
                meta={"file_name": file_name, "batch_id": batch_id},
            )
            for file_name, data in images
        ]
        return queue.enqueue_many(jobs)

    def load_batch(batch_id: str) -> dict:
        raw = redis.get(f"batch:{batch_id}")
        if raw is None:
            raise HTTPException(404, "Batch not found (unknown ID or expired)")
        return json.loads(raw)

    @app.get("/health")
    def health():
        try:
            redis.ping()
        except RedisError:
            return JSONResponse({"status": "error", "redis": False}, status_code=503)
        return {
            "status": "ok",
            "redis": True,
            "queue_depth": len(queue),
            "workers": Worker.count(queue=queue),
            "model": config.MODEL_NAME,
        }

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile):
        data = await file.read()
        validate_image(data)
        [job] = enqueue_images([(file.filename or "upload", data)])
        return {"job_id": job.id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = fetch_job(job_id)
        status = job.get_status().value  # queued / started / finished / failed / ...
        body = {"job_id": job.id, "status": status, "result": None, "error": None}
        if status == "finished":
            body["result"] = job.return_value()
        elif status == "failed":
            body["error"] = job_error(job)
        return body

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str, min_score: float = Query(0.0, ge=0.0, le=1.0)):
        """Just the model output (COCO-style JSON for the segmentation model).

        ``min_score`` drops annotations below that confidence; the default (0)
        returns every detection the worker stored.
        """
        job = fetch_job(job_id)
        status = job.get_status().value
        if status != "finished":
            raise HTTPException(409, f"Job is not finished yet (status: {status})")
        result = {k: v for k, v in job.return_value().items() if k not in JOB_EXTRA_KEYS}
        return coco.filter_by_score(result, min_score) if coco.is_coco(result) else result

    @app.get("/jobs/{job_id}/preview")
    def get_job_preview(job_id: str):
        """The image with boxes drawn on it (only for models that make a preview)."""
        job = fetch_job(job_id)
        # Build the path from the ID RQ gave us, never from raw user input.
        path = config.OUTPUT_DIR / job.id / "preview.png"
        if not path.exists():
            raise HTTPException(404, "No preview for this job (not finished, or model has none)")
        return FileResponse(path, media_type="image/png")

    @app.post("/batches", status_code=202)
    async def create_batch(files: list[UploadFile]):
        """Accept several images at once and create one job per image."""
        images = []
        for file in files:
            data = await file.read()
            name = file.filename or f"image_{len(images) + 1}"
            validate_image(data, name)  # one bad file rejects the whole batch
            images.append((name, data))

        batch_id = uuid.uuid4().hex
        jobs = enqueue_images(images, batch_id)  # raises 429 if the batch doesn't fit

        # A batch is just a list of job IDs stored in Redis. It expires together
        # with the job results, so nothing has to clean it up.
        record = {
            "created_at": time.time(),
            "jobs": [
                {"job_id": job.id, "file_name": name}
                for job, (name, _data) in zip(jobs, images, strict=True)
            ],
        }
        redis.set(f"batch:{batch_id}", json.dumps(record), ex=config.RESULT_TTL_SECONDS)
        return {"batch_id": batch_id, "total": len(jobs), "job_ids": [job.id for job in jobs]}

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: str):
        """Progress of a batch: counts plus the status of every job in it."""
        entries = load_batch(batch_id)["jobs"]
        # fetch_many loads all jobs in one round trip; expired jobs come back as None.
        jobs = Job.fetch_many([e["job_id"] for e in entries], connection=redis)

        statuses = []
        for entry, job in zip(entries, jobs, strict=True):
            status = job.get_status().value if job else "expired"
            error = job_error(job) if status == "failed" else None
            statuses.append({**entry, "status": status, "error": error})

        completed = sum(s["status"] == "finished" for s in statuses)
        failed = sum(s["status"] in ("failed", "expired") for s in statuses)
        return {
            "batch_id": batch_id,
            "total": len(entries),
            "completed": completed,
            "failed": failed,
            "pending": len(entries) - completed - failed,
            "done": completed + failed == len(entries),
            "jobs": statuses,
        }

    @app.get("/batches/{batch_id}/export")
    def export_batch(batch_id: str, min_score: float = Query(0.0, ge=0.0, le=1.0)):
        """All finished jobs of a batch merged into one COCO JSON file (a download)."""
        entries = load_batch(batch_id)["jobs"]
        jobs = Job.fetch_many([e["job_id"] for e in entries], connection=redis)

        items = []
        for entry, job in zip(entries, jobs, strict=True):
            if job is None or job.get_status().value != "finished":
                continue  # failed or unfinished images are left out
            result = job.return_value()
            if not coco.is_coco(result):
                raise HTTPException(
                    400,
                    "This model doesn't produce COCO output; use MODEL_NAME=deeplabv3_mobilenet",
                )
            items.append((entry["job_id"], entry["file_name"], result))
        if not items:
            raise HTTPException(409, "No finished jobs in this batch yet")

        disposition = f'attachment; filename="batch-{batch_id}.coco.json"'
        return JSONResponse(
            coco.merge(items, min_score), headers={"Content-Disposition": disposition}
        )

    # A plain "def" endpoint: FastAPI runs it in a thread pool, so the slow,
    # CPU-bound model call does not block the async event loop.
    @app.post("/predict_sync")
    def predict_sync(file: UploadFile):
        start = time.perf_counter()
        data = file.file.read()
        validate_image(data)
        _image, prediction, inference_seconds = predict_image(data)
        metrics.observe(redis, metrics.SYNC_LATENCY, time.perf_counter() - start)
        return {**prediction.result, "inference_ms": round(inference_seconds * 1000, 2)}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
