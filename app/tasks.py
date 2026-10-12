"""The function that workers run for every job.

The API puts ``run_inference(image_bytes, submitted_at)`` on the RQ queue. RQ
stores the function name and its arguments in Redis; a worker process picks
the job up, calls this function, and stores the return value back in Redis
as the job's result.
"""

import io
import json
import time

from PIL import Image
from rq import get_current_job

from app import config, metrics
from app.models import Prediction, get_model


def decode_image(image_bytes: bytes) -> Image.Image:
    """Bytes from the upload -> RGB PIL image. Raises if the bytes are not an image."""
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def predict_image(image_bytes: bytes) -> tuple[Prediction, float]:
    """Decode an image and run the model. Returns (prediction, seconds spent in the model)."""
    image = decode_image(image_bytes)
    start = time.perf_counter()
    prediction = get_model().predict(image)
    return prediction, time.perf_counter() - start


def save_outputs(job_id: str, prediction: Prediction) -> dict:
    """Write the result JSON (and preview image, if any) to OUTPUT_DIR/<job_id>/."""
    job_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    files = {"json": str(job_dir / "result.json")}
    (job_dir / "result.json").write_text(json.dumps(prediction.result, indent=2))

    if prediction.preview is not None:
        Image.fromarray(prediction.preview).save(job_dir / "preview.png")
        files["preview"] = str(job_dir / "preview.png")
    return files


def run_inference(image_bytes: bytes, submitted_at: float) -> dict:
    """Entry point executed by the worker for one queued job."""
    job = get_current_job()
    redis = job.connection
    try:
        prediction, inference_seconds = predict_image(image_bytes)
        output_files = save_outputs(job.id, prediction)
    except Exception:
        metrics.increment(redis, metrics.JOBS_FAILED)
        raise  # re-raise so RQ marks the job as "failed" and stores the traceback

    # time.time() is wall-clock time, so it can be compared across processes
    # on the same machine (the API recorded submitted_at when it enqueued).
    total_seconds = time.time() - submitted_at
    metrics.increment(redis, metrics.JOBS_COMPLETED)
    metrics.observe(redis, metrics.INFERENCE_LATENCY, inference_seconds)
    metrics.observe(redis, metrics.JOB_LATENCY, total_seconds)

    return {
        **prediction.result,
        "inference_ms": round(inference_seconds * 1000, 2),
        "output_files": output_files,
    }
