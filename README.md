# Provenance ESG Processing Pipeline

**Version 1.0.0** · Apache-2.0 · in production on EuroHPC Leonardo Booster since 2026

Batch OCR and layout parsing for corporate sustainability, ESG and annual reports. The pipeline turns long, table-heavy PDF reports into structured Markdown, HTML and JSON with page-level provenance, as the input for ESG data extraction on [SustainabilityReports.com](https://www.sustainabilityreports.com).

It runs the open-weights [Chandra OCR 2](https://huggingface.co/datalab-to/chandra-ocr-2) vision-language model on [vLLM](https://github.com/vllm-project/vllm), one server per GPU, and is built for SLURM clusters: each job is one 4-GPU node working through a folder of about 300 reports.

## At a glance

| | |
|---|---|
| Documents processed on Leonardo | more than 110,000 reports (2026) |
| Throughput, one Booster node (4x A100 64 GB) | ~6,500 pages per hour (2.22 s per page per GPU) |
| Cost at production scale | 0.7 GPU hours per 1,000 pages, failures and re-runs included |
| Output per report | `<report>_full.json` (pages, layout blocks, bounding boxes, Markdown, HTML) |

Measurements and how they were taken: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## How it works

```
 campaign scheduler (our servers)                       Leonardo Booster node (SLURM job)
 ───────────────────────────────                        ─────────────────────────────────────────────
 select reports ─▶ budget guard ─▶ stage PDFs ─▶ sbatch ─▶ 4x vLLM servers (Singularity, 1 per GPU)
                     (campaign/)    (sha256)                   ▲
                                                               │ HTTP, batched pages
                                                       scripts/run_pipeline.py
                                                         render pages (pypdfium2, 200 DPI)
                                                         → OCR → merge → <report>_full.json
                                                         heartbeat watchdog, timeout retries
 harvest ◀───────────────────────────────────────────── outputs ◀┘
 validate, render Markdown, extract ESG data
```

1. **Campaign planning.** Reports are grouped into folders sized to finish inside one job's wall time (fixed warm-up plus a per-page allowance).
2. **Budget guard** ([campaign/budget_guard.py](campaign/budget_guard.py)). Before every submission: `remaining - unsettled - estimate >= floor`, plus a linearised monthly quota so a three-month allocation is spread evenly. HPC accounting lags the scheduler by a day or more, so not-yet-billed jobs are counted explicitly.
3. **Node job** ([hpc/leonardo/chandra_ocr_node.slurm](hpc/leonardo/chandra_ocr_node.slurm)). Starts one vLLM server per GPU from the official container, waits for health checks and continues with whichever GPUs came up (graceful degradation).
4. **Worker** ([scripts/run_pipeline.py](scripts/run_pipeline.py)). Renders pages, sends them in batches to the healthy endpoints, retries over-long pages at reduced resolution, merges batches into one `_full.json` per report and skips reports that are already done, so any job can be re-run safely.
5. **Watchdog.** A log-heartbeat watchdog cancels a job only when the worker has been silent for 15 minutes, so a genuinely hung vLLM server does not burn the rest of the wall time. Timed-out batches are retried twice before the job ends.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/run_pipeline.py` | Multi-endpoint OCR worker (production version) |
| `hpc/leonardo/chandra_ocr_node.slurm` | One-node SLURM job: vLLM servers, worker, watchdog, retries |
| `campaign/budget_guard.py` | Allocation and monthly-quota checks, wall-time sizing |
| `chandra/` | Chandra OCR library by Datalab (Apache-2.0), with a bounding-box fix (see [NOTICE.md](NOTICE.md)) |
| `tests/` | Unit tests |
| `docs/BENCHMARKS.md` | Throughput and cost measurements |

## Running it

### On a single machine

Start a vLLM server with the model, then point the worker at it:

```bash
pip install -e .
vllm serve datalab-to/chandra-ocr-2 --served-model-name chandra \
    --dtype bfloat16 --max-model-len 18000 --port 8000
```

`config.json`:

```json
{
  "pdf_path": "input/",
  "endpoints": {"gpu-0": "localhost:8000"},
  "processing": {"workers_per_gpu": 12, "fresh_restart_interval": 0}
}
```

```bash
OCR_DIR=output/ IMAGE_DPI=200 MIN_IMAGE_DIM=2048 python scripts/run_pipeline.py config.json
```

Add one entry to `endpoints` per GPU or server; the worker balances across them and drops endpoints that fail repeatedly.

### On Leonardo (or another SLURM cluster)

```bash
sbatch --account=<ACCOUNT> -J ocr_batch_01 hpc/leonardo/chandra_ocr_node.slurm /path/to/pdf/folder
```

The job expects the vLLM container (`vllm/vllm-openai:v0.17.0` as a Singularity image), a host virtual environment with this repository installed and an offline Hugging Face cache of the model under `$CHANDRA_WORK_DIR` (default `$WORK`). See the header of the SLURM file.

### Output

```
output/
└── XNAS_AAPL_2024_SR/
    └── XNAS_AAPL_2024_SR_full.json   # pages, blocks with bounding boxes, Markdown, HTML, token counts
```

## Tests

```bash
pip install pytest
pytest
```

## Acknowledgements

We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy), through the AI Factory Playground and Fast Lane access calls (EHPC-AIF-2025PG01-531, EHPC-AIF-2026PG01-109, EHPC-AIF-2026FL01-210, EHPC-AIF-2026PG01-959).

- **Chandra OCR** by [Datalab](https://www.datalab.to/): OCR model and inference library
- **vLLM**: high-throughput inference serving
- **pypdfium2**: PDF parsing and rendering

## License

Apache License 2.0, see [LICENSE](LICENSE). The Chandra OCR model weights are under a modified OpenRAIL-M licence; see [NOTICE.md](NOTICE.md).

## Contact

SustainabilityReports.com · [hello@sustainabilityreports.com](mailto:hello@sustainabilityreports.com) · [github.com/SustainabilityReports](https://github.com/SustainabilityReports)
