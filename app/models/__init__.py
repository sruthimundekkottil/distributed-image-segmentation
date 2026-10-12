"""Model registry: maps a MODEL_NAME to the class that implements it."""

from app import config
from app.models.base import InferenceModel, Prediction
from app.models.classifier import MobileNetV2Classifier

# Add new models here. The key is the value you put in MODEL_NAME.
MODEL_REGISTRY: dict[str, type[InferenceModel]] = {
    "mobilenet_v2": MobileNetV2Classifier,
}

# One loaded model per process. Loading is slow, so we only do it once.
_loaded_model: InferenceModel | None = None


def load_model(name: str) -> InferenceModel:
    """Create a new model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown MODEL_NAME {name!r}. Choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config.MODEL_DIR, num_threads=config.ORT_NUM_THREADS)


def get_model() -> InferenceModel:
    """Return this process's model, loading it the first time it is needed."""
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = load_model(config.MODEL_NAME)
    return _loaded_model


def set_model(model: InferenceModel | None) -> None:
    """Replace the process-wide model (used by tests to inject a fake model)."""
    global _loaded_model
    _loaded_model = model


__all__ = ["InferenceModel", "Prediction", "MODEL_REGISTRY", "load_model", "get_model", "set_model"]
