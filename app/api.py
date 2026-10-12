"""HTTP API.

Two ways to get a prediction:

* Asynchronous (the "production" path):
    POST /jobs        -> puts the image on a Redis queue, returns a job ID immediately
    GET  /jobs/{id}   -> poll until status is "finished", then read the result
  Separate worker processes do the actual inference.

* Synchronous baseline (for comparison):
    POST /predict_sync -> runs the model inside the API process and waits for it

Run locally with:  uvicorn app.api:app --port 8000
"""

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app import config, metrics
from app.models import get_model
from app.tasks import predict_image

log = logging.getLogger("api")


def validate_image(data: bytes) -> None:
    """Fail fast with HTTP 400/413 instead of letting a worker choke on a bad upload."""
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {config.MAX_UPLOAD_BYTES} bytes")
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(400, "Upload is not a valid image") from exc


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

        # Backpressure: if the workers are already far behind, refuse new work now
        # instead of accepting it and making every client wait longer and longer.
        # (The check and the enqueue are two separate steps, so under heavy
        # concurrency the queue can overshoot the limit slightly. That's fine here.)
        depth = len(queue)
        if depth >= config.MAX_QUEUE_DEPTH:
            metrics.increment(redis, metrics.JOBS_REJECTED)
            raise HTTPException(
                429,
                f"Queue is full ({depth} jobs waiting). Try again later.",
                headers={"Retry-After": "1"},
            )

        # We enqueue the function by its import path, so the API never needs to
        # import worker-only code. time.time() lets the worker measure queue wait.
        job = queue.enqueue(
            "app.tasks.run_inference",
            data,
            time.time(),
            result_ttl=config.RESULT_TTL_SECONDS,
            failure_ttl=config.RESULT_TTL_SECONDS,
        )
        return {"job_id": job.id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            job = Job.fetch(job_id, connection=redis)
        except NoSuchJobError as exc:
            raise HTTPException(404, "Job not found (unknown ID or result expired)") from exc

        status = job.get_status().value  # queued / started / finished / failed / ...
        body = {"job_id": job.id, "status": status, "result": None, "error": None}
        if status == "finished":
            body["result"] = job.return_value()
        elif status == "failed":
            result = job.latest_result()
            # Only return the last line of the traceback (e.g. "ValueError: ...").
            body["error"] = result.exc_string.strip().splitlines()[-1] if result else "failed"
        return body

    # A plain "def" endpoint: FastAPI runs it in a thread pool, so the slow,
    # CPU-bound model call does not block the async event loop.
    @app.post("/predict_sync")
    def predict_sync(file: UploadFile):
        start = time.perf_counter()
        data = file.file.read()
        validate_image(data)
        prediction, inference_seconds = predict_image(data)
        metrics.observe(redis, metrics.SYNC_LATENCY, time.perf_counter() - start)
        return {**prediction.result, "inference_ms": round(inference_seconds * 1000, 2)}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
