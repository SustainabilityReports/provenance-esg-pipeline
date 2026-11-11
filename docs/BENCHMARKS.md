# Benchmarks

All numbers are from production runs on Leonardo Booster (CINECA): one node = 4x NVIDIA A100 64 GB, 32 cores. Accounting: 1 GPU hour = 8 local core hours.

## Single-node throughput (18 April 2026)

One 1-node job, 500 integrated reports, Chandra OCR 2 on vLLM, 200 DPI.

| Metric | Per GPU | Per node |
|---|---|---|
| Pages per hour | 1,625 | 6,500 |
| Seconds per page | 2.22 | |
| Reports per hour (mean 171 pages) | 9.5 | 38 |

- 452 of 500 reports finished (77,306 pages) inside a 12-hour wall time; the job ran out of wall time, not throughput.
- Warm-up before the first output (vLLM start and model load): about 15 minutes.
- Page counts per report: median 141, p90 336, maximum 1,407.

Sizing rule that follows: about 400 reports (or 76,000 pages) per 12-hour one-node job.

## Cost per document across allocations

Effective end-to-end rates: accounting (`saldo`) core hours divided by documents converted. Failed jobs, re-runs and warm-up are included, so these are what a campaign actually costs, not a controlled benchmark.

| Allocation | Model | Documents | GPU hours per document |
|---|---|---|---|
| EHPC-AIF-2026PG01-109 (Feb-Apr 2026) | Chandra OCR 1 | 2,996 | 1.84 |
| EHPC-AIF-2026FL01-210 (Apr-Jul 2026) | Chandra OCR 2, tuned vLLM batching | 50,312 | 0.19 |
| EHPC-AIF-2026PG01-959 (Aug-Nov 2026) | Chandra OCR 2 | ~60,000 (7.1M pages) | ~0.08 |

Moving from Chandra OCR 1 to Chandra OCR 2 with tuned batching cut the cost per document about tenfold. At production scale the pipeline uses **0.7 GPU hours per 1,000 pages**. Per-document figures also depend on the document mix (report length varies from a few pages to over 1,000).

## vLLM settings used

```
--dtype bfloat16 --max-model-len 18000 --max-num-seqs 64
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85
--enable-prefix-caching --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
```

Worker: batch size 4 pages, 12 concurrent requests per GPU, maximum 16,000 output tokens per page, pages rendered at 200 DPI with a 2,048-pixel minimum dimension.
