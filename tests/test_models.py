import numpy as np
import pytest
from PIL import Image

from app import metrics
from app.models import load_model
from app.models.classifier import preprocess, softmax


def test_preprocess_shape_and_normalisation():
    batch = preprocess(Image.new("RGB", (640, 480), color=(124, 116, 104)))
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == np.float32
    # (124,116,104)/255 is almost exactly the ImageNet mean, so this is ~0 after normalising.
    assert np.abs(batch).max() < 0.02


def test_softmax_sums_to_one():
    probabilities = softmax(np.array([1.0, 2.0, 3.0]))
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.argmax() == 2


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown MODEL_NAME"):
        load_model("not-a-model")


def test_histogram_buckets_are_cumulative(redis):
    metrics.observe(redis, metrics.JOB_LATENCY, 0.03)
    metrics.observe(redis, metrics.JOB_LATENCY, 0.7)
    raw = redis.hgetall(metrics.KEY_PREFIX + metrics.JOB_LATENCY)
    assert raw.get(b"0.025") is None  # 0.03 is above this bucket
    assert raw[b"0.05"] == b"1"  # only the 0.03 observation
    assert raw[b"1.0"] == b"2"  # both observations
    assert raw[b"+Inf"] == b"2"
    assert float(raw[b"sum"]) == pytest.approx(0.73)
