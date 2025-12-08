# Provenance ESG Processing Pipeline

**Version 0.1-alpha**

A production-ready pipeline for extracting structured data from corporate ESG and sustainability reports at scale. Built for high-throughput batch processing on HPC/AI infrastructure.

## Overview

This pipeline ingests large batches of corporate ESG/sustainability reports (PDF), performs OCR and layout parsing, extracts text and images while preserving document structure, and outputs standardized formats (Markdown, HTML, JSON) suitable for downstream NLP analysis.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PROVENANCE ESG PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐  │
│   │  PDFs   │───▶│  PDF Parser  │───▶│  9B-param   │───▶│ Structured│  │
│   │ (batch) │    │  pypdfium2   │    │  VLM/OCR    │    │  Output   │  │
│   └─────────┘    └──────────────┘    └─────────────┘    └───────────┘  │
│        │                │                   │                 │        │
│        │                │                   │                 │        │
│        ▼                ▼                   ▼                 ▼        │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐  │
│   │  File   │    │    Image     │    │   Batch     │    │ Markdown  │  │
│   │Validate │    │  Rendering   │    │  Inference  │    │ HTML/JSON │  │
│   └─────────┘    └──────────────┘    └─────────────┘    └───────────┘  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │              ProcessPoolExecutor (parallel files)               │  │
│   │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    Endpoint Failover         │  │
│   │   │GPU 0│ │GPU 1│ │GPU 2│ │GPU N│    Health Monitoring         │  │
│   │   └─────┘ └─────┘ └─────┘ └─────┘    Auto-retry on failure      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Batch Processing**: Concurrent PDF processing with configurable parallelism
- **Multi-GPU Support**: Distributes workload across available GPU endpoints
- **Endpoint Failover**: Automatic detection and routing around failed endpoints (spot instance tolerant)
- **Robust Error Handling**: File validation, retry logic, graceful degradation
- **Structured Output**: Markdown, HTML, and JSON with page-level metadata
- **Image Extraction**: Preserves figures, charts, and diagrams with bbox coordinates
- **Layout Preservation**: Maintains document structure (headers, footers, tables)

## Requirements

- Python 3.10+
- CUDA-capable GPU(s) with 24GB+ VRAM (for local inference)
- vLLM server endpoint (for distributed inference)

### Dependencies

```
pypdfium2>=4.30.0      # PDF parsing and rendering
torch>=2.8.0           # Deep learning framework
transformers>=4.57.1   # Model inference
openai>=2.2.0          # vLLM API client
pillow>=10.2.0         # Image processing
beautifulsoup4>=4.14.2 # HTML parsing
pydantic>=2.12.0       # Data validation
```

## Installation

```bash
git clone https://github.com/SustainabilityReports/provenance-esg-pipeline.git
cd provenance-esg-pipeline
pip install -e .
```

## Usage

### Configuration

Set environment variables or create a `.env` file:

```bash
# vLLM endpoint configuration
VLLM_API_BASE=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL_NAME=chandra

# Processing settings
IMAGE_DPI=200
MIN_IMAGE_DIM=1536
```

### Running the Pipeline

```bash
# Process all PDFs in input directory
python scripts/run_pipeline.py

# Or import as module
from scripts.run_pipeline import main
main()
```

### Output Structure

Each processed PDF creates a subdirectory:

```
output_reports/
└── Company_2024_Report/
    ├── Company_2024_Report.md           # Markdown output
    ├── Company_2024_Report.html         # HTML output
    ├── Company_2024_Report_metadata.json # Page metadata
    ├── Company_2024_Report_full.json    # Full structured output
    └── images/
        ├── abc123_1_img.webp            # Extracted figures
        └── ...
```

## Scalability & HPC Readiness

### Parallelization Strategy

- **File-level parallelism**: ProcessPoolExecutor isolates PDF processing
- **Batch inference**: Pages grouped into batches (configurable size)
- **Multi-endpoint**: Workload distributed across GPU endpoints
- **Configurable concurrency**: `FILES_PER_GPU` parameter for tuning

### Resource Requirements (Estimated)

| Workload | GPUs | Time | Pages/Hour |
|----------|------|------|------------|
| 100 reports (~10K pages) | 1x A100 | ~8 hours | 1,250 |
| 1,000 reports (~100K pages) | 4x A100 | ~20 hours | 5,000 |
| 10,000 reports (~1M pages) | 16x A100 | ~50 hours | 20,000 |

### Containerization

The pipeline is designed for containerization (Docker/Singularity):

```dockerfile
FROM nvidia/cuda:12.1-base
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python", "scripts/run_pipeline.py"]
```

## Architecture

### Core Components

1. **Input Stage** (`chandra/input.py`): PDF loading, page rendering
2. **Inference Stage** (`chandra/model/`): VLM inference via vLLM or HuggingFace
3. **Output Stage** (`chandra/output.py`): HTML/Markdown conversion, image extraction
4. **Orchestration** (`scripts/run_pipeline.py`): Parallel processing, failover, logging

### OCR Model

Uses the Chandra 9B-parameter vision-language model optimized for document understanding:
- Supports 40+ languages
- Accurate table reconstruction
- Handwriting recognition
- Math/formula extraction

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The underlying Chandra OCR model weights are subject to a modified OpenRAIL-M license (see [NOTICE.md](NOTICE.md)).

## Acknowledgments

- **Chandra OCR** by [Datalab](https://www.datalab.to/) - Core OCR model and inference library
- **vLLM** - High-throughput inference serving
- **pypdfium2** - PDF parsing and rendering

## Contact

Sustainability Reports B.V.  
GitHub: [SustainabilityReports](https://github.com/SustainabilityReports)

