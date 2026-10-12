"""The small interface every model in this service implements.

The worker and the API never talk to ONNX Runtime directly. They only call
``model.predict(image)``. To add a new model you:

1. write a class that subclasses ``InferenceModel`` and implements ``predict``
2. register it in ``app/models/__init__.py``
3. select it with the ``MODEL_NAME`` environment variable
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class Prediction:
    """What a model returns for one image."""

    # JSON-serialisable result that is sent back to the client and saved to disk.
    result: dict
    # Optional visualisation (an RGB image as a numpy array, HxWx3, uint8).
    # The worker saves it as a PNG next to the JSON if it is present.
    preview: np.ndarray | None = None


class InferenceModel(ABC):
    """Base class for all models."""

    # Short identifier, e.g. "mobilenet_v2". Used in results and file names.
    name: str = "base"

    @abstractmethod
    def predict(self, image: Image.Image) -> Prediction:
        """Run the model on one RGB PIL image."""


def create_onnx_session(model_path: Path, num_threads: int) -> ort.InferenceSession:
    """Load an .onnx file into an ONNX Runtime session that runs on the CPU.

    Loading is slow (hundreds of milliseconds), so each process does it once
    at startup and then reuses the session for every request.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run the export script in scripts/ first.")
    options = ort.SessionOptions()
    if num_threads > 0:
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
