"""MobileNetV2 image classifier (ImageNet, 1000 classes) running on ONNX Runtime."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.models.base import InferenceModel, Prediction, create_onnx_session

# The mean/std the network was trained with. Inputs must be normalised the same way.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """Turn a PIL image into the 1x3x224x224 float tensor MobileNetV2 expects.

    This copies torchvision's standard ImageNet transform:
    resize the short side to 256, centre-crop 224x224, scale to [0, 1],
    then normalise with the ImageNet mean/std.
    """
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)

    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    image = image.crop((left, top, left + size, top + size))

    array = np.asarray(image, dtype=np.float32) / 255.0  # HxWxC in [0, 1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return array[np.newaxis, ...]  # add batch dimension -> 1xCxHxW


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)  # for numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


class MobileNetV2Classifier(InferenceModel):
    name = "mobilenet_v2"

    def __init__(self, model_dir: Path, num_threads: int = 1, top_k: int = 5):
        self.session = create_onnx_session(model_dir / "mobilenet_v2.onnx", num_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.labels = json.loads((model_dir / "imagenet_classes.json").read_text())
        self.top_k = top_k

    def predict(self, image: Image.Image) -> Prediction:
        batch = preprocess(image)
        logits = self.session.run(None, {self.input_name: batch})[0][0]  # shape (1000,)
        probabilities = softmax(logits)
        top = np.argsort(probabilities)[::-1][: self.top_k]
        predictions = [
            {
                "class_id": int(i),
                "label": self.labels[i],
                "score": round(float(probabilities[i]), 4),
            }
            for i in top
        ]
        return Prediction(result={"model": self.name, "predictions": predictions})
