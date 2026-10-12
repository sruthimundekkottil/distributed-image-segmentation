"""Tests for the segmentation post-processing (no real ONNX model needed)."""

import json

import numpy as np
from PIL import Image

from app import config
from app.models import set_model
from app.models.segmenter import INPUT_SIZE, DeepLabV3Segmenter, find_objects, mask_to_polygons
from tests.test_api import upload

CLASS_NAMES = ["__background__"] + [f"class{i}" for i in range(1, 21)]
DOG, CAT, PERSON = 12, 8, 15


class FakeSession:
    """Pretends to be an onnxruntime.InferenceSession.

    It returns logits for a 520x520 image with two dog blobs, one cat blob,
    and one low-confidence "person" blob that should be filtered out.
    """

    def get_inputs(self):
        class Input:
            name = "input"

        return [Input()]

    def run(self, _outputs, _feeds):
        logits = np.zeros((1, 21, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        logits[0, 0] = 5.0  # background everywhere ...
        logits[0, DOG, 50:150, 50:250] = 10.0  # ... except a dog,
        logits[0, DOG, 300:400, 300:400] = 10.0  # a second, separate dog,
        logits[0, CAT, 400:500, 20:120] = 10.0  # and a cat.
        # Barely beats background: softmax confidence ~0.59, below min_score=0.6.
        logits[0, PERSON, 10:60, 400:500] = 5.5
        return [logits]


def make_segmenter() -> DeepLabV3Segmenter:
    # Skip __init__ (which loads the .onnx file) and plug in the fakes instead.
    segmenter = DeepLabV3Segmenter.__new__(DeepLabV3Segmenter)
    segmenter.session = FakeSession()
    segmenter.input_name = "input"
    segmenter.class_names = CLASS_NAMES
    segmenter.min_area_fraction = 0.001
    segmenter.min_score = 0.6
    return segmenter


def test_find_objects_splits_blobs_and_ignores_background_and_noise():
    labels = np.zeros((100, 100), dtype=np.uint8)
    labels[10:20, 10:30] = DOG  # 200 px
    labels[50:60, 50:60] = DOG  # 100 px, separate blob
    labels[90, 90] = CAT  # 1 px of noise
    confidence = np.full((100, 100), 0.8, dtype=np.float32)

    objects = find_objects(labels, confidence, min_area=5)

    assert [(c, bbox) for c, _, bbox, _ in objects] == [
        (DOG, [10, 10, 20, 10]),
        (DOG, [50, 50, 10, 10]),
    ]
    assert all(abs(score - 0.8) < 1e-6 for *_, score in objects)


def test_mask_to_polygons_traces_rectangle():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 5:25] = True
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1
    xs, ys = polygons[0][0::2], polygons[0][1::2]
    assert (min(xs), max(xs), min(ys), max(ys)) == (5, 24, 10, 19)


def test_predict_returns_coco_and_preview_in_original_coordinates():
    # Original image is 1040x520, i.e. 2x wider than the model input,
    # so x coordinates must be scaled back up by 2.
    image = Image.new("RGB", (1040, 520), color=(30, 30, 30))
    prediction = make_segmenter().predict(image)
    coco = prediction.result

    assert coco["images"] == [{"id": 1, "file_name": "input.jpg", "width": 1040, "height": 520}]
    assert len(coco["categories"]) == 20
    assert coco["info"]["preview_min_score"] == 0.6
    annotations = coco["annotations"]
    # Two dogs, a cat, and the low-confidence "person": the JSON keeps everything
    # so clients can pick their own threshold.
    names = sorted(a["category_name"] for a in annotations)
    assert names == ["class12", "class12", "class15", "class8"]
    person = next(a for a in annotations if a["category_name"] == "class15")
    assert 0.55 < person["score"] < 0.6

    first_dog = next(a for a in annotations if a["bbox"][1] == 50)
    assert first_dog["bbox"] == [100, 50, 400, 100]  # x and width doubled
    assert first_dog["area"] == 400 * 100
    assert first_dog["iscrowd"] == 0
    assert 0.99 < first_dog["score"] <= 1.0
    assert len(first_dog["segmentation"][0]) >= 6  # at least 3 (x, y) points

    assert prediction.preview.shape == (520, 1040, 3)
    assert prediction.preview.dtype == np.uint8
    background = (30, 30, 30)
    # The confident dog was drawn (pixel inside its mask changed) ...
    assert tuple(prediction.preview[100, 300]) != background
    # ... but the "person" below min_score was not (pixel inside it is untouched).
    # Person blob: rows 10-60, columns 400-500 at 520px -> 800-1000 at 1040px.
    assert tuple(prediction.preview[35, 900]) == background
    json.dumps(coco)  # must be JSON-serialisable


def test_worker_saves_coco_json_preview_and_input(client, png_bytes, run_worker):
    set_model(make_segmenter())
    job_id = client.post("/jobs", files=upload(png_bytes)).json()["job_id"]
    run_worker()

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "finished"
    job_dir = config.OUTPUT_DIR / job_id
    saved = json.loads((job_dir / "result.json").read_text())
    assert saved["annotations"] and saved["images"][0]["file_name"] == "input.jpg"
    assert (job_dir / "input.jpg").exists()

    response = client.get(f"/jobs/{job_id}/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_preview_404_for_model_without_preview(client, png_bytes, run_worker, fake_model):
    fake_model.make_preview = False
    job_id = client.post("/jobs", files=upload(png_bytes)).json()["job_id"]
    run_worker()
    assert client.get(f"/jobs/{job_id}/preview").status_code == 404
