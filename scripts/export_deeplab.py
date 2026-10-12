"""Export a pretrained torchvision DeepLabV3-MobileNetV3 segmentation model to ONNX.

Usage:  python scripts/export_deeplab.py [--out-dir models]

Writes:
  models/deeplabv3_mobilenet.onnx  the network (input 1x3x520x520, output 1x21x520x520 logits)
  models/voc_classes.json          the 21 class names (index 0 is "__background__")

The model was trained on a subset of COCO using the 20 Pascal VOC categories
(person, car, dog, cat, ...), so those are the only objects it can segment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

INPUT_SIZE = 520  # the resolution torchvision's pretrained weights were evaluated at


class OutputOnly(torch.nn.Module):
    """torchvision segmentation models return a dict ({"out": ..., "aux": ...}).
    ONNX wants plain tensors, so this wrapper returns just the main output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "deeplabv3_mobilenet.onnx"

    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = OutputOnly(deeplabv3_mobilenet_v3_large(weights=weights)).eval()

    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamo=True,  # the modern torch.export-based exporter
        verbose=False,
        external_data=False,  # keep the weights inside the single .onnx file
    )

    categories = weights.meta["categories"]
    (args.out_dir / "voc_classes.json").write_text(json.dumps(categories))

    # Sanity check: ONNX Runtime should give the same answer as PyTorch.
    with torch.no_grad():
        expected = model(example).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": example.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Wrote {onnx_path} ({size_mb:.1f} MB); max |torch - onnx| = {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX output does not match PyTorch output"


if __name__ == "__main__":
    main()
