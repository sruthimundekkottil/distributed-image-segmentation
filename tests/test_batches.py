"""Tests for POST /batches, GET /batches/{id}, /batches/{id}/export and /jobs/{id}/result."""

from app import coco, config, metrics
from app.models import set_model
from tests.conftest import FakeModel
from tests.test_api import upload
from tests.test_segmenter import make_segmenter


def batch_upload(png_bytes: bytes, count: int) -> list:
    return [("files", (f"img{i}.png", png_bytes, "image/png")) for i in range(count)]


def queue_depth(client) -> int:
    return client.get("/health").json()["queue_depth"]


def test_create_batch_enqueues_one_job_per_image(client, png_bytes):
    response = client.post("/batches", files=batch_upload(png_bytes, 3))
    assert response.status_code == 202
    body = response.json()
    assert body["total"] == 3
    assert len(set(body["job_ids"])) == 3
    assert queue_depth(client) == 3


def test_batch_status_before_and_after_processing(client, png_bytes, run_worker):
    batch_id = client.post("/batches", files=batch_upload(png_bytes, 3)).json()["batch_id"]

    status = client.get(f"/batches/{batch_id}").json()
    assert (status["total"], status["completed"], status["failed"]) == (3, 0, 0)
    assert status["pending"] == 3 and status["done"] is False
    assert [job["status"] for job in status["jobs"]] == ["queued"] * 3
    assert [job["file_name"] for job in status["jobs"]] == ["img0.png", "img1.png", "img2.png"]

    run_worker()

    status = client.get(f"/batches/{batch_id}").json()
    assert (status["completed"], status["failed"], status["pending"]) == (3, 0, 0)
    assert status["done"] is True


def test_batch_reports_failed_jobs_with_error(client, png_bytes, run_worker):
    set_model(FakeModel(fail=True))
    batch_id = client.post("/batches", files=batch_upload(png_bytes, 2)).json()["batch_id"]
    run_worker()

    status = client.get(f"/batches/{batch_id}").json()
    assert (status["completed"], status["failed"], status["done"]) == (0, 2, True)
    assert all("model exploded" in job["error"] for job in status["jobs"])


def test_too_large_batch_is_rejected_with_429_and_nothing_enqueued(
    client, redis, png_bytes, monkeypatch
):
    monkeypatch.setattr(config, "MAX_QUEUE_DEPTH", 4)
    client.post("/jobs", files=upload(png_bytes))
    client.post("/jobs", files=upload(png_bytes))

    # 2 waiting + 3 new = 5 > 4: the whole batch is refused, not just the overflow.
    response = client.post("/batches", files=batch_upload(png_bytes, 3))
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"
    assert "smaller batch" in response.json()["detail"]
    assert queue_depth(client) == 2
    assert int(redis.get(metrics.KEY_PREFIX + metrics.JOBS_REJECTED)) == 3

    # A batch that fits exactly is still accepted.
    assert client.post("/batches", files=batch_upload(png_bytes, 2)).status_code == 202
    assert queue_depth(client) == 4


def test_batch_larger_than_queue_limit_is_rejected_even_when_queue_empty(
    client, png_bytes, monkeypatch
):
    monkeypatch.setattr(config, "MAX_QUEUE_DEPTH", 2)
    assert client.post("/batches", files=batch_upload(png_bytes, 3)).status_code == 429
    assert queue_depth(client) == 0


def test_batch_with_one_invalid_image_is_rejected(client, png_bytes):
    files = batch_upload(png_bytes, 2) + [("files", ("notes.txt", b"hello", "text/plain"))]
    response = client.post("/batches", files=files)
    assert response.status_code == 400
    assert "notes.txt" in response.json()["detail"]
    assert queue_depth(client) == 0


def test_unknown_batch_returns_404(client):
    assert client.get("/batches/nope").status_code == 404
    assert client.get("/batches/nope/export").status_code == 404


def test_job_result_returns_coco_and_filters_by_score(client, png_bytes, run_worker):
    set_model(make_segmenter())
    job_id = client.post("/jobs", files=upload(png_bytes)).json()["job_id"]
    assert client.get(f"/jobs/{job_id}/result").status_code == 409  # not finished yet

    run_worker()

    result = client.get(f"/jobs/{job_id}/result").json()
    assert "inference_ms" not in result and "output_files" not in result
    # Every detection is stored, including the low-confidence "person" (~0.6 here).
    assert len(result["annotations"]) == 4
    filtered = client.get(f"/jobs/{job_id}/result", params={"min_score": 0.7}).json()
    assert len(filtered["annotations"]) == 3
    assert client.get(f"/jobs/{job_id}/result", params={"min_score": 2}).status_code == 422


def test_batch_export_merges_into_one_coco_file(client, png_bytes, run_worker):
    set_model(make_segmenter())
    batch = client.post("/batches", files=batch_upload(png_bytes, 2)).json()
    assert client.get(f"/batches/{batch['batch_id']}/export").status_code == 409  # nothing done
    run_worker()

    response = client.get(f"/batches/{batch['batch_id']}/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    merged = response.json()

    assert [(i["id"], i["file_name"], i["job_id"]) for i in merged["images"]] == [
        (1, "img0.png", batch["job_ids"][0]),
        (2, "img1.png", batch["job_ids"][1]),
    ]
    assert len(merged["categories"]) == 20
    annotations = merged["annotations"]
    assert len(annotations) == 8  # 4 objects per image
    assert [a["id"] for a in annotations] == list(range(1, 9))  # unique, renumbered
    assert [a["image_id"] for a in annotations] == [1] * 4 + [2] * 4

    filtered = client.get(f"/batches/{batch['batch_id']}/export", params={"min_score": 0.7})
    assert len(filtered.json()["annotations"]) == 6


def test_batch_export_needs_a_coco_model(client, png_bytes, run_worker):
    batch_id = client.post("/batches", files=batch_upload(png_bytes, 1)).json()["batch_id"]
    run_worker()  # FakeModel returns a classification-style result, not COCO
    response = client.get(f"/batches/{batch_id}/export")
    assert response.status_code == 400
    assert "deeplabv3_mobilenet" in response.json()["detail"]


def test_merge_renumbers_ids_and_filters():
    single = {
        "images": [{"id": 1, "file_name": "input.jpg", "width": 10, "height": 8}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "score": 0.9},
            {"id": 2, "image_id": 1, "category_id": 1, "score": 0.3},
        ],
    }
    merged = coco.merge([("job-a", "a.jpg", single), ("job-b", "b.jpg", single)], min_score=0.5)
    assert [i["file_name"] for i in merged["images"]] == ["a.jpg", "b.jpg"]
    assert [(a["id"], a["image_id"]) for a in merged["annotations"]] == [(1, 1), (2, 2)]
    assert len(single["annotations"]) == 2  # the input was not modified
