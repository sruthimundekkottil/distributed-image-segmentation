"""Export a pretrained torchvision MobileNetV2 to ONNX.

Usage:  python scripts/export_mobilenet.py [--out-dir models]

Writes:
  models/mobilenet_v2.onnx       the network, runnable with ONNX Runtime (no PyTorch needed)
  models/imagenet_classes.json   the 1000 ImageNet class names, in output order

PyTorch is only needed for this one-off export step. The API and workers only
need the small onnxruntime package, which keeps their Docker image lean.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "mobilenet_v2.onnx"

    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights).eval()  # eval(): turn off dropout / batchnorm updates

    # ONNX export works by tracing the model on an example input. We use a fixed
    # shape of one 224x224 RGB image, which is what the worker always sends.
    example = torch.randn(1, 3, 224, 224)
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
    (args.out_dir / "imagenet_classes.json").write_text(json.dumps(categories))

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
