"""Locust load test.

Two kinds of simulated users. Pick one by naming it on the command line:

    locust -f loadtest/locustfile.py --host http://localhost:8000 AsyncJobUser
    locust -f loadtest/locustfile.py --host http://localhost:8000 SyncUser

AsyncJobUser  submits an image to POST /jobs, then polls GET /jobs/{id} until
              the job is done. The full submit -> result time is reported as a
              custom "JOB end_to_end" entry, which is the number that matters
              to a real client.
SyncUser      calls POST /predict_sync and waits for the answer.

Both users send the next request as soon as the previous one finishes
(no think time), so N users means N requests in flight at all times.
"""

import os
import time
from pathlib import Path

from locust import HttpUser, constant, task

IMAGE_PATH = Path(os.environ.get("LOAD_TEST_IMAGE", Path(__file__).parent / "sample.jpg"))
IMAGE_BYTES = IMAGE_PATH.read_bytes()

# How often an AsyncJobUser asks "is my job done yet?". Shorter = more accurate
# latency numbers, but more GET requests hitting the API.
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "0.02"))
JOB_TIMEOUT_SECONDS = 120


def image_upload() -> dict:
    return {"file": (IMAGE_PATH.name, IMAGE_BYTES, "image/jpeg")}


class AsyncJobUser(HttpUser):
    wait_time = constant(0)

    @task
    def submit_and_wait(self):
        start = time.perf_counter()

        with self.client.post(
            "/jobs", files=image_upload(), name="POST /jobs", catch_response=True
        ) as response:
            if response.status_code == 429:
                # Backpressure kicked in. Count it as a failure so it shows up in
                # the stats, back off briefly, and try again on the next task.
                response.failure("429 queue full")
                time.sleep(1)
                return
            if response.status_code != 202:
                response.failure(f"unexpected status {response.status_code}")
                return
            job_id = response.json()["job_id"]

        error = None
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
            status = self.client.get(f"/jobs/{job_id}", name="GET /jobs/[id]").json()["status"]
            if status == "finished":
                break
            if status == "failed":
                error = RuntimeError("job failed")
                break
            if time.perf_counter() - start > JOB_TIMEOUT_SECONDS:
                error = TimeoutError("job did not finish in time")
                break

        # Report the whole round trip to Locust as its own "request" type.
        self.environment.events.request.fire(
            request_type="JOB",
            name="end_to_end",
            response_time=(time.perf_counter() - start) * 1000,
            response_length=0,
            exception=error,
            context={},
        )


class SyncUser(HttpUser):
    wait_time = constant(0)

    @task
    def predict(self):
        self.client.post("/predict_sync", files=image_upload(), name="POST /predict_sync")
