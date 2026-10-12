"""Prometheus metrics that are shared between the API and the workers.

The problem: the API and each worker are separate processes (often separate
containers). A normal ``prometheus_client.Counter`` lives in the memory of one
process, so a counter incremented in a worker would never show up on the
API's ``/metrics`` page.

The simple fix used here: workers write their numbers into Redis (which
everyone can already reach), and the API reads them back from Redis every
time Prometheus scrapes ``/metrics``.

(The production alternatives are prometheus_client's "multiprocess mode", a
Pushgateway, or letting Prometheus scrape every worker directly.)
"""

from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
from prometheus_client.registry import Collector
from redis import Redis
from rq import Queue, Worker

KEY_PREFIX = "metrics:"

# Upper bounds (in seconds) of the latency histogram buckets.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

# Counter names (stored as plain Redis integers).
JOBS_COMPLETED = "jobs_completed_total"
JOBS_FAILED = "jobs_failed_total"
JOBS_REJECTED = "jobs_rejected_total"

# Histogram names (stored as Redis hashes).
JOB_LATENCY = "job_latency_seconds"  # enqueue -> result ready (includes queue wait)
INFERENCE_LATENCY = "model_seconds"  # time a worker spends inside model.predict
SYNC_LATENCY = "predict_sync_latency_seconds"  # /predict_sync request handling time

HISTOGRAM_HELP = {
    JOB_LATENCY: "Time from enqueue until the job result is ready (queue wait + inference).",
    INFERENCE_LATENCY: "Time a worker spends inside model.predict for one image.",
    SYNC_LATENCY: "Time to serve one POST /predict_sync request.",
}


def increment(redis: Redis, counter: str, amount: int = 1) -> None:
    redis.incrby(KEY_PREFIX + counter, amount)


def observe(redis: Redis, histogram: str, seconds: float) -> None:
    """Record one observation in a histogram stored as a Redis hash.

    Prometheus histograms are *cumulative*: the bucket ``le=0.5`` counts every
    observation <= 0.5s. So we increment every bucket the value fits into,
    plus the ``+Inf`` bucket, the running ``sum`` and the ``count``.
    A pipeline sends all the commands to Redis in one round trip.
    """
    key = KEY_PREFIX + histogram
    pipe = redis.pipeline()
    for upper_bound in LATENCY_BUCKETS:
        if seconds <= upper_bound:
            pipe.hincrby(key, str(upper_bound), 1)
    pipe.hincrby(key, "+Inf", 1)
    pipe.hincrbyfloat(key, "sum", seconds)
    pipe.hincrby(key, "count", 1)
    pipe.execute()


class RedisMetricsCollector(Collector):
    """Builds Prometheus metrics from Redis each time /metrics is scraped."""

    def __init__(self, redis: Redis, queue: Queue):
        self.redis = redis
        self.queue = queue

    def collect(self):
        depth = GaugeMetricFamily("inference_queue_depth", "Jobs waiting in the queue.")
        depth.add_metric([], len(self.queue))
        yield depth

        workers = GaugeMetricFamily("inference_workers", "Workers listening on the queue.")
        workers.add_metric([], Worker.count(queue=self.queue))
        yield workers

        for name, help_text in [
            (JOBS_COMPLETED, "Jobs that finished successfully."),
            (JOBS_FAILED, "Jobs that raised an error."),
            (JOBS_REJECTED, "Requests rejected with HTTP 429 because the queue was full."),
        ]:
            # CounterMetricFamily adds the "_total" suffix itself.
            counter = CounterMetricFamily("inference_" + name.removesuffix("_total"), help_text)
            counter.add_metric([], int(self.redis.get(KEY_PREFIX + name) or 0))
            yield counter

        for name, help_text in HISTOGRAM_HELP.items():
            yield self._histogram(name, help_text)

    def _histogram(self, name: str, help_text: str) -> HistogramMetricFamily:
        raw = {k.decode(): v.decode() for k, v in self.redis.hgetall(KEY_PREFIX + name).items()}
        buckets = [(str(b), int(raw.get(str(b), 0))) for b in LATENCY_BUCKETS]
        buckets.append(("+Inf", int(raw.get("+Inf", 0))))
        histogram = HistogramMetricFamily("inference_" + name, help_text)
        histogram.add_metric([], buckets, sum_value=float(raw.get("sum", 0)))
        return histogram
