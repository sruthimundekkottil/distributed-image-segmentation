# Benchmark results

- Date: 2026-10-12 20:36
- **CPU cores: 4 physical / 8 logical on the host, 8 available to Docker**
- CPU: AMD Ryzen 5 7235HS (Windows 11)
- Model: `mobilenet_v2` on ONNX Runtime (CPU), `ORT_NUM_THREADS=1` per process, `API_WORKERS=4` uvicorn process(es)
- Load: 16 concurrent Locust users, no think time, 60 s per scenario, same JPEG (loadtest/sample.jpg) every time
- Locust, the API, Redis and the workers all ran on the same machine.

| Scenario | Workers | Images completed | Throughput (images/s) | p50 latency (ms) | p95 latency (ms) | Failed | Rejected (429) |
|---|---|---|---|---|---|---|---|
| Queue + 4 workers | 4 | 1461 | 24.6 | 610 | 920 | 0 | 0 |

Latency for the queue scenarios is the full client-side round trip: upload, wait in the queue, inference, and polling `GET /jobs/{id}` every 20 ms until the result is ready.
