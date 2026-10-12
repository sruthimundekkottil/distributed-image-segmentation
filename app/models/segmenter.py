"""DeepLabV3-MobileNetV3 semantic segmentation, turned into per-object boxes.

DeepLabV3 is a *semantic* segmentation model: for every pixel it predicts a
class ("dog", "person", ...), but it does not tell two dogs apart. To get
individual objects we post-process the mask with OpenCV:

1. softmax over the 21 class scores -> best class and its confidence per pixel
2. for each class, split its mask into connected blobs (one blob = one object)
3. for each blob: bounding box, polygon outline, area, mean confidence
4. write the objects in COCO format and draw them on a preview image

Limitation: two touching objects of the same class become one blob.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.models.base import (
    INPUT_IMAGE_FILENAME,
    InferenceModel,
    Prediction,
    create_onnx_session,
)
from app.models.classifier import IMAGENET_MEAN, IMAGENET_STD, softmax

INPUT_SIZE = 520  # must match scripts/export_deeplab.py

# One bright colour (RGB) per class for drawing; index = class id.
# Hues are spread evenly around the colour wheel so neighbouring classes differ.
PALETTE = [
    tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])
    for h in np.linspace(0, 179, 21, endpoint=False).astype(int)
]


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL image -> normalised 1x3x520x520 float tensor.

    The model was exported with a fixed 520x520 input, so we stretch the image
    to a square. The mask is resized back to the original size afterwards.
    """
    resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return array.transpose(2, 0, 1)[np.newaxis, ...]


def find_objects(
    labels: np.ndarray, confidence: np.ndarray, min_area: int
) -> list[tuple[int, np.ndarray, list[int], float]]:
    """Split a per-pixel label map into separate objects.

    Returns a list of (class_id, object_mask, [x, y, w, h], mean_confidence).
    """
    objects = []
    for class_id in np.unique(labels):
        if class_id == 0:  # 0 is background
            continue
        class_mask = (labels == class_id).astype(np.uint8)
        # Label each separate blob of this class with its own number 1..count-1.
        count, blob_ids, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for blob in range(1, count):  # blob 0 is "not this class"
            x, y, w, h, area = stats[blob]
            if area < min_area:  # ignore specks of noise
                continue
            mask = blob_ids == blob
            score = float(confidence[mask].mean())
            objects.append((int(class_id), mask, [int(x), int(y), int(w), int(h)], score))
    return objects


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """Outline of a binary mask as COCO polygons: [[x1, y1, x2, y2, ...], ...]."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)  # drop redundant points
        if len(contour) >= 3:  # a polygon needs at least 3 points
            polygons.append(contour.reshape(-1).astype(float).tolist())
    return polygons


def draw_preview(image: np.ndarray, objects, class_names: list[str]) -> np.ndarray:
    """Draw a translucent mask, a bounding box and a label for every object."""
    preview = image.copy()
    overlay = image.copy()
    for class_id, mask, _, _ in objects:
        overlay[mask] = PALETTE[class_id]
    preview = cv2.addWeighted(overlay, 0.4, preview, 0.6, 0)

    thickness = max(2, round(max(image.shape[:2]) / 400))  # scale lines with image size
    for class_id, _, (x, y, w, h), score in objects:
        color = PALETTE[class_id]
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, thickness)
        text = f"{class_names[class_id]} {score:.2f}"
        font_scale = thickness / 3
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(y - th - 6, 0)
        cv2.rectangle(preview, (x, top), (x + tw + 4, top + th + 6), color, -1)
        cv2.putText(
            preview, text, (x + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )  # fmt: skip
    return preview


class DeepLabV3Segmenter(InferenceModel):
    name = "deeplabv3_mobilenet"

    def __init__(
        self,
        model_dir: Path,
        num_threads: int = 1,
        min_area_fraction: float = 0.001,
        min_score: float = 0.6,
    ):
        self.session = create_onnx_session(model_dir / "deeplabv3_mobilenet.onnx", num_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = json.loads((model_dir / "voc_classes.json").read_text())
        # Blobs smaller than this fraction of the image are treated as noise.
        self.min_area_fraction = min_area_fraction
        # Objects whose mean confidence is below this are dropped. Low-confidence
        # blobs are usually the model being unsure (e.g. patches of a white dog's
        # fur labelled "cat" at ~0.55). 0.6 was picked by eye, not tuned on data.
        self.min_score = min_score

    def predict(self, image: Image.Image) -> Prediction:
        image = image.convert("RGB")
        width, height = image.size

        logits = self.session.run(None, {self.input_name: preprocess(image)})[0][0]  # 21xHxW
        probabilities = softmax(logits, axis=0)
        labels = probabilities.argmax(axis=0).astype(np.uint8)
        confidence = probabilities.max(axis=0)

        # Back to the original resolution so boxes are in original pixel coordinates.
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)

        min_area = max(1, int(self.min_area_fraction * width * height))
        objects = [
            obj for obj in find_objects(labels, confidence, min_area) if obj[3] >= self.min_score
        ]
        coco = self.to_coco(objects, width, height)
        preview = draw_preview(np.asarray(image), objects, self.class_names)
        return Prediction(result=coco, preview=preview)

    def to_coco(self, objects, width: int, height: int) -> dict:
        """Build a COCO-style dict (the format used by most detection tools).

        bbox is [x, y, width, height] in pixels, segmentation is a list of
        polygons, and score is the mean per-pixel confidence of the object.
        """
        annotations = [
            {
                "id": i,
                "image_id": 1,
                "category_id": class_id,
                "category_name": self.class_names[class_id],
                "bbox": bbox,
                "area": int(mask.sum()),
                "segmentation": mask_to_polygons(mask),
                "iscrowd": 0,
                "score": round(score, 4),
            }
            for i, (class_id, mask, bbox, score) in enumerate(objects, start=1)
        ]
        return {
            "model": self.name,
            "images": [
                {"id": 1, "file_name": INPUT_IMAGE_FILENAME, "width": width, "height": height}
            ],
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(self.class_names) if i > 0
            ],
            "annotations": annotations,
        }
