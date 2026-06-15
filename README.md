# AUGUS

AUGUS is a training-free framework for user-level privacy leakage inference from public social media posts. It treats privacy profiling as an evidence-driven investigation: perceive post clues, maintain hypotheses, route tools/models for evidence collection, verify evidence, aggregate across posts with a Cross-Post Evidence Graph (CPEG), and project an evidence-grounded profile.

Paper: [From Posts to Profiles: User-Level Privacy Leakage on Social Media](https://arxiv.org/abs/2606.06784)

This repository contains the core AUGUS/ARGUS implementation and the baseline runners only. It does not include private datasets, generated outputs, API keys, or local server paths.

## What Is Included

- `argus/`: main pipeline, CPEG, hypothesis state, routing, verification, tool adapters, model adapters.
- `baselines/`: text-only, post-wise VLM aggregation, single-agent, self-disclosure detector, and HolmesEye-adapted baselines.
- `scripts/argus/run_json_argus.sh`: run AUGUS on user-level JSON inputs.
- `scripts/argus/run_json_baseline.sh`: run baselines on the same input format.
- `examples/sample_user.json`: minimal fictional input example.

## Install

```bash
git clone git@github.com:sfofgalaxy/AUGUS.git
cd AUGUS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your own model and tool credentials. Do not commit `.env`.

For OCR-heavy runs, install your local PaddleOCR-VL environment separately and point `ARGUS_TOOL_DEVICE` / `CUDA_VISIBLE_DEVICES` to the device you want to use.

## Input Format

AUGUS expects one JSON file per user, or a directory containing user JSON files.

```json
{
  "user_id": "sample_user_001",
  "platform": "synthetic",
  "metadata": {
    "nickname": "Sample User",
    "bio": "Weekend runner",
    "ip_location": "California"
  },
  "posts": [
    {
      "post_id": "post_001",
      "title": "Morning routine",
      "caption": "Early run near campus.",
      "tags": ["running", "campus"],
      "timestamp": "2026-01-10T08:30:00Z",
      "media_files": ["path/to/image.jpg"],
      "metadata": {
        "location": null
      }
    }
  ]
}
```

`media_files` can contain existing absolute paths, paths relative to the repository/current working directory, HTTP(S) URLs, or image data URLs.

## Run AUGUS

```bash
INPUT=examples/sample_user.json \
OUTPUT_DIR=outputs/argus/sample \
bash scripts/argus/run_json_argus.sh
```

Useful options:

```bash
GPU=0 INPUT=inputs/users OUTPUT_DIR=outputs/argus/run1 bash scripts/argus/run_json_argus.sh
LIMIT=5 INPUT=inputs/users OUTPUT_DIR=outputs/argus/debug bash scripts/argus/run_json_argus.sh
```

Outputs are written under `OUTPUT_DIR/<user_id>/`, including `profile.json`, `cpeg.json`, `step_logs.json`, and `metrics.json`.

## Run Baselines

All baselines use the same JSON input format.

```bash
BASELINE=text_only INPUT=examples/sample_user.json OUTPUT_DIR=outputs/baselines/text_only \
bash scripts/argus/run_json_baseline.sh

BASELINE=single_post INPUT=examples/sample_user.json OUTPUT_DIR=outputs/baselines/single_post \
bash scripts/argus/run_json_baseline.sh

BASELINE=single_agent INPUT=examples/sample_user.json OUTPUT_DIR=outputs/baselines/single_agent \
bash scripts/argus/run_json_baseline.sh

BASELINE=self_disclosure_detector INPUT=examples/sample_user.json OUTPUT_DIR=outputs/baselines/self_disclosure \
bash scripts/argus/run_json_baseline.sh

BASELINE=holmeseye_adapted INPUT=examples/sample_user.json OUTPUT_DIR=outputs/baselines/holmeseye \
bash scripts/argus/run_json_baseline.sh
```

Supported baseline names:

- `text_only`: direct user-level LLM inference from all post text.
- `single_post`: post-wise VLM/text analysis followed by profile aggregation.
- `single_agent`: one tool-using agent with the same public inputs and tools, without AUGUS verifier/CPEG state management.
- `self_disclosure_detector`: post-level self-disclosure detection and aggregation.
- `holmeseye_adapted`: HolmesEye-style visual cue extraction adapted to social media posts.

Model overrides:

```bash
MODEL_PROVIDER=qwen MODEL=qwen3.7-max BASELINE=text_only INPUT=inputs/users OUTPUT_DIR=outputs/baselines/qwen_text \
bash scripts/argus/run_json_baseline.sh
```

## Environment Variables

The main OpenAI-compatible backbone is configured with:

- `ARGUS_ROUTER_BASE_URL`
- `ARGUS_ROUTER_API_KEY`
- `ARGUS_ROUTER_MODEL`
- `ARGUS_INVESTIGATOR_MODEL`
- `ARGUS_VERIFIER_MODEL`
- `ARGUS_NARRATIVE_MODEL`

Qwen/DashScope:

- `DASHSCOPE_API_KEY`
- `ARGUS_QWEN_TEXT_MODEL` defaults to `qwen3.7-max`
- `ARGUS_QWEN_VL_MODEL` defaults to `qwen3.6-plus`

Gemini/Vertex:

- `GOOGLE_KEY_PATH`
- `GOOGLE_CLOUD_LOCATION`
- `ARGUS_GEMINI_MODEL`

Optional tools:

- `SERPAPI_API_KEY`
- `AMAP_API_KEY`
- `GOOGLE_MAPS_API_KEY`

## Citation

If you use AUGUS, please cite:

```bibtex
@misc{augus2026posts,
  title = {From Posts to Profiles: User-Level Privacy Leakage on Social Media},
  year = {2026},
  eprint = {2606.06784},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2606.06784}
}
```

## Safety Note

This code is intended for privacy research and defensive analysis. Do not use it to identify, harass, contact, or profile real people without proper authorization and ethical review.
