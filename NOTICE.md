# NOTICE

This project includes software developed by third parties.

## Chandra OCR

The `chandra/` directory contains code from the Chandra OCR project by Datalab (Endless Labs, Inc.).

- **Source**: https://github.com/datalab-to/chandra
- **License**: Apache License 2.0
- **Copyright**: Copyright 2025 Endless Labs, Inc.

### Code License (Apache 2.0)

The Chandra codebase is licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

### Model License (Modified OpenRAIL-M)

The Chandra model weights (`datalab-to/chandra` and `datalab-to/chandra-ocr-2` on Hugging Face) are subject to a modified OpenRAIL-M license:

- **Free for**: Research, personal use, and startups under $2M funding/revenue
- **Restrictions**: Cannot be used competitively with Datalab's API
- **Commercial licensing**: Available at https://www.datalab.to/pricing

For the full model license, see: https://github.com/datalab-to/chandra/blob/master/MODEL_LICENSE

## Modifications

This project includes modifications to the original Chandra code:
- Bounding box Y-axis correction in `chandra/output.py` (BBOX_Y_OFFSET parameter)
- Custom batch processing orchestration in `scripts/run_pipeline.py`, `hpc/leonardo/` and `campaign/` (original work of this project)

## Other Dependencies

This project also depends on the following open source projects:

- **vLLM** (Apache 2.0): https://github.com/vllm-project/vllm (run from the official `vllm/vllm-openai` container)
- **pypdfium2** (Apache 2.0 / BSD): https://github.com/ArtifexSoftware/pdfium
- **HuggingFace Transformers** (Apache 2.0): https://github.com/huggingface/transformers
- **PyTorch** (BSD-style): https://github.com/pytorch/pytorch

