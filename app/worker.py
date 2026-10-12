"""Worker process: loads the model once, then processes jobs from Redis forever.

Run with:  python -m app.worker

Why ``SimpleWorker`` and not RQ's default ``Worker``?
The default worker forks a brand-new child process for every job. That is
great for isolation, but for ML it means:
  * the model would have to be reloaded (or at least re-touched) per job, and
  * ONNX Runtime's thread pool does not survive ``fork()`` safely.
``SimpleWorker`` runs jobs inside this same process, so the model loaded
below stays in memory and is reused for every job. To use more CPU cores
you start more worker processes (``docker compose up --scale worker=4``).
"""

import logging
import time

from PIL import Image
from redis import Redis
from rq import Queue, SimpleWorker

from app import config
from app.models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def main() -> None:
    start = time.perf_counter()
    model = get_model()  # load the ONNX model before accepting any jobs
    # The first inference call is slower (memory allocation, kernel selection),
    # so run one dummy image now instead of making the first real user wait.
    model.predict(Image.new("RGB", (320, 240)))
    log.info("Loaded and warmed up %r in %.2fs", model.name, time.perf_counter() - start)

    redis = Redis.from_url(config.REDIS_URL)
    queue = Queue(config.QUEUE_NAME, connection=redis)
    worker = SimpleWorker([queue], connection=redis)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
