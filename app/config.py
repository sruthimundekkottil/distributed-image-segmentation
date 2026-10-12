"""All runtime settings live here and come from environment variables.

Keeping configuration in one place makes it easy to see every knob the
service exposes, and environment variables are what docker-compose (and
Kubernetes, etc.) use to configure containers.
"""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Where Redis lives. Inside docker-compose the hostname is the service name.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Name of the RQ queue that the API pushes to and the workers pull from.
QUEUE_NAME = os.environ.get("QUEUE_NAME", "inference")

# Backpressure: once this many jobs are waiting, new requests get HTTP 429.
MAX_QUEUE_DEPTH = _int("MAX_QUEUE_DEPTH", 100)

# How long (seconds) finished job results are kept in Redis.
RESULT_TTL_SECONDS = _int("RESULT_TTL_SECONDS", 3600)

# Reject uploads larger than this many bytes.
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)

# Which model to serve. See app/models/__init__.py for the available names.
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenet_v2")

# Folder containing the exported .onnx files (created by scripts/export_*.py).
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))

# Folder where workers write per-job output files (JSON + preview image).
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

# Number of CPU threads ONNX Runtime may use *inside one inference call*.
# With several worker containers on one machine, giving each worker 1 thread
# avoids them fighting over the same cores. 0 means "let ONNX Runtime decide"
# (usually: all cores).
ORT_NUM_THREADS = _int("ORT_NUM_THREADS", 1)
