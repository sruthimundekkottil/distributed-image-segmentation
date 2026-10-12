import json
from pathlib import Path

from app import config, metrics
from app.models import set_model
from tests.conftest import FakeModel


def upload(png_bytes: bytes) -> dict:
    return {"file": ("test.png", png_bytes, "image/png")}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["queue_depth"] == 0


def test_create_job_returns_id_and_enqueues(client, png_bytes):
    response = client.post("/jobs", files=upload(png_bytes))
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert client.get("/health").json()["queue_depth"] == 1


def test_create_job_rejects_non_image(client):
    response = client.post("/jobs", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_create_job_rejects_too_large_upload(client, png_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/jobs", files=upload(png_bytes))
    assert response.status_code == 413


def test_backpressure_returns_429_when_queue_full(client, redis, png_bytes, monkeypatch):
    monkeypatch.setattr(config, "MAX_QUEUE_DEPTH", 2)
    assert client.post("/jobs", files=upload(png_bytes)).status_code == 202
    assert client.post("/jobs", files=upload(png_bytes)).status_code == 202

    response = client.post("/jobs", files=upload(png_bytes))
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert int(redis.get(metrics.KEY_PREFIX + metrics.JOBS_REJECTED)) == 1


def test_job_lifecycle_queued_to_finished(client, png_bytes, run_worker):
    job_id = client.post("/jobs", files=upload(png_bytes)).json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"

    run_worker()

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "finished"
    assert body["result"]["predictions"][0]["label"] == "cat"
    assert body["result"]["inference_ms"] >= 0

    # The worker saved the result JSON and preview image to disk.
    files = body["result"]["output_files"]
    assert json.loads(Path(files["json"]).read_text())["model"] == "fake"
    assert Path(files["preview"]).exists()


def test_failed_job_reports_error(client, redis, png_bytes, run_worker):
    set_model(FakeModel(fail=True))
    job_id = client.post("/jobs", files=upload(png_bytes)).json()["job_id"]
    run_worker()

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert "model exploded" in body["error"]
    assert int(redis.get(metrics.KEY_PREFIX + metrics.JOBS_FAILED)) == 1


def test_unknown_job_returns_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_predict_sync(client, png_bytes):
    response = client.post("/predict_sync", files=upload(png_bytes))
    assert response.status_code == 200
    assert response.json()["predictions"][0]["label"] == "cat"


def test_metrics_endpoint(client, png_bytes, run_worker):
    client.post("/jobs", files=upload(png_bytes))
    client.post("/jobs", files=upload(png_bytes))
    assert "inference_queue_depth 2.0" in client.get("/metrics").text

    run_worker()
    text = client.get("/metrics").text
    assert "inference_queue_depth 0.0" in text
    assert "inference_jobs_completed_total 2.0" in text
    assert "inference_job_latency_seconds_count 2.0" in text
    assert 'inference_job_latency_seconds_bucket{le="+Inf"} 2.0' in text


def test_frontend_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert '<script src="/static/app.js">' in page.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert client.get(asset).status_code == 200
