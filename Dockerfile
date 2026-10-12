# ---------------------------------------------------------------------------
# Stage 1: "exporter" - uses PyTorch to download the pretrained models and
# export them to ONNX. PyTorch is ~1 GB, so this stage is thrown away and
# only the resulting .onnx files are copied into the final image.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS exporter
WORKDIR /build
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements-export.txt .
RUN pip install --no-cache-dir -r requirements-export.txt
COPY scripts/export_mobilenet.py scripts/
RUN python scripts/export_mobilenet.py --out-dir /build/models
COPY scripts/export_deeplab.py scripts/
RUN python scripts/export_deeplab.py --out-dir /build/models

# ---------------------------------------------------------------------------
# Stage 2: runtime image shared by the api and worker services.
# Only needs onnxruntime (no PyTorch), so it stays small.
# ---------------------------------------------------------------------------
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=exporter /build/models/ models/
COPY app/ app/
ENV MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/outputs \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# The api service uses this default command; the worker service overrides it.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
