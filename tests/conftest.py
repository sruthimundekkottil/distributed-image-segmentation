"""Shared test fixtures.

The tests never need a real Redis server or a real ONNX model:
* fakeredis gives us an in-memory Redis that RQ can talk to, and
* FakeModel stands in for the neural network so tests run in milliseconds.
"""

import io

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from rq import Queue, SimpleWorker
from rq.timeouts import TimerDeathPenalty

from app import config
from app.api import create_app
from app.models import InferenceModel, Prediction, set_model


class FakeModel(InferenceModel):
    name = "fake"

    def __init__(self, fail: bool = False, make_preview: bool = True):
        self.fail = fail
        self.make_preview = make_preview

    def predict(self, image: Image.Image) -> Prediction:
        if self.fail:
            raise ValueError("model exploded")
        preview = None
        if self.make_preview:
            preview = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        return Prediction(
            result={"model": self.name, "predictions": [{"label": "cat", "score": 0.9}]},
            preview=preview,
        )


@pytest.fixture(autouse=True)
def fake_model(tmp_path, monkeypatch):
    """Every test uses the fake model and writes outputs to a temp folder."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    model = FakeModel()
    set_model(model)
    yield model
    set_model(None)


@pytest.fixture
def redis():
    return fakeredis.FakeRedis()


@pytest.fixture
def client(redis):
    return TestClient(create_app(redis))


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def run_worker(redis):
    """Process every queued job once, the same way app/worker.py does, then return."""

    def _run():
        queue = Queue(config.QUEUE_NAME, connection=redis)
        worker = SimpleWorker([queue], connection=redis)
        # The default timeout mechanism uses SIGALRM, which doesn't exist on
        # Windows; a timer thread works everywhere.
        worker.death_penalty_class = TimerDeathPenalty
        worker.work(burst=True)

    return _run
