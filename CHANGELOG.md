# Changelog

## 1.0.0 (2026-11-11)

Production version as run on EuroHPC Leonardo Booster.

- OCR model: Chandra OCR 2 (`datalab-to/chandra-ocr-2`) served by vLLM 0.17 from the official container, one server per GPU.
- `scripts/run_pipeline.py`: production worker. Static multi-endpoint configuration from a JSON file, health checks and endpoint failover, reduced-resolution retry for pages that exceed the model length, per-report merge into `_full.json`, idempotent re-runs, stall detection.
- `hpc/leonardo/chandra_ocr_node.slurm`: one-node SLURM job with graceful degradation when a GPU fails to start, a log-heartbeat watchdog and two retry rounds for timed-out batches.
- `campaign/budget_guard.py`: allocation check (`remaining - unsettled - estimate >= floor`), linearised monthly quota and wall-time sizing, with unit tests.
- `docs/BENCHMARKS.md`: measured throughput and cost per document.
- Continuous integration: tests and a byte-compile check on every push.

## 0.1.0-alpha (2025-12-08)

Initial release: Chandra OCR library, batch pipeline with multi-GPU support and endpoint failover, Markdown/HTML/JSON output.
