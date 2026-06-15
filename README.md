# AUGUS

AUGUS is a training-free framework for user-level privacy leakage inference from public social media posts. It treats privacy profiling as an evidence-driven investigation: perceive post clues, maintain hypotheses, route tools/models for evidence collection, verify evidence, aggregate across posts with a Cross-Post Evidence Graph (CPEG), and project an evidence-grounded profile.

Paper: [From Posts to Profiles: User-Level Privacy Leakage on Social Media](https://arxiv.org/abs/2606.06784)

This repository contains the core AUGUS/ARGUS implementation, baseline runners, tool adapters, input converters, and local preprocessing utilities. It does not include private datasets, generated outputs, API keys, or local server paths.

## What Is Included

- `argus/`: main pipeline, CPEG, hypothesis state, routing, verification, tool adapters, model adapters.
- `argus/skill_adapters/`: OCR, crop/zoom, web search, map search, and webpage fetching tools.
- `baselines/`: text-only, post-wise VLM aggregation, single-agent, self-disclosure detector, and HolmesEye-adapted baselines.
- `scripts/argus/run_json_argus.sh`: run AUGUS on user-level JSON inputs.
- `scripts/argus/run_json_baseline.sh`: run baselines on the same input format.
- `scripts/argus/convert_*.py`: convert common dataset exports into unified AUGUS JSON.
- `scripts/argus/extract_video_frames.py`: turn local videos into representative image frames.
- `scripts/argus/test_local_tools.sh`: smoke-test local tools and optional network/OCR tools.
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

Optional OCR:

```bash
pip install -r requirements-ocr.txt
```

For OCR-heavy runs, point `ARGUS_TOOL_DEVICE` / `CUDA_VISIBLE_DEVICES` to the device you want to use.

## Configure

Start from `.env.example`. At minimum, set:

```bash
ARGUS_ROUTER_BASE_URL=https://your-openai-compatible-host/v1
ARGUS_ROUTER_API_KEY=<your-router-api-key>
ARGUS_ROUTER_MODEL=gpt-5.5
DASHSCOPE_API_KEY=<your-dashscope-api-key>
```

Set these only if you use the corresponding tools:

```bash
GOOGLE_KEY_PATH=/absolute/path/to/google-service-account.json
SERPAPI_API_KEY=<your-serpapi-key>
AMAP_API_KEY=<your-amap-key>
```

Important defaults already shown in `.env.example`:

- `ARGUS_QWEN_TEXT_MODEL=qwen3.7-max`
- `ARGUS_QWEN_VL_MODEL=qwen3.6-plus`
- `ARGUS_QWEN_PERCEPTION_MAX_SIDE=768`
- `ARGUS_TOOL_DEVICE=auto`
- `ARGUS_CACHE_DIR=outputs/argus/cache`
- `ARGUS_LLM_MAX_RETRIES=5`
- `ARGUS_GEMINI_MAX_RETRIES=7`

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

`media_files` can contain existing absolute paths, paths relative to the JSON file/repository/current working directory, HTTP(S) URLs, or image data URLs.

The main pipeline consumes images. If your raw JSON contains videos, run the video-frame preprocessing step below first.

## Convert Inputs

Xiaohongshu txt exports:

```bash
python scripts/argus/convert_xhs_txt_to_json.py \
  --input-dir path/to/xhs/user_notes \
  --output-dir inputs/argus/json/xhs
```

Instagram folder exports:

```bash
python scripts/argus/convert_ins_json_to_json.py \
  --input-dir path/to/instagram/all \
  --output-dir inputs/argus/json/ins
```

Synthetic post/image exports:

```bash
python scripts/argus/convert_synthetic_json_to_json.py \
  --post-dir path/to/Synthetic_new/outputs/post \
  --image-dir path/to/Synthetic_new/final_img \
  --output-dir inputs/argus/json/synthetic
```

Video preprocessing:

```bash
python scripts/argus/extract_video_frames.py \
  --input inputs/argus/json/ins \
  --output-dir inputs/argus/json/ins_frames \
  --frames-per-video 3 \
  --max-side 768
```

After this, run AUGUS on `inputs/argus/json/ins_frames`.

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

Output layout:

- `profile.json`: final evidence-grounded user profile.
- `cpeg.json`: Cross-Post Evidence Graph.
- `step_logs.json`: investigator/verifier trace per post.
- `metrics.json`: timing, token, image, and tool-call metrics.
- `[[ERROR]]` entries: written when a user run fails and `ARGUS_CONTINUE_ON_ERROR=1`.

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

Google web search and Google Maps search both use SerpApi in this implementation.

## Tools and Skills

The runtime tools live in `argus/skill_adapters/`. Each tool has a `SKILL.md` and optional `references/` notes.

| Tool | Function | Required config |
|---|---|---|
| `run_ocr` | PaddleOCR-VL OCR for screenshots, signs, tickets, receipts, documents, small text | `requirements-ocr.txt`, `ARGUS_TOOL_DEVICE` |
| `crop_image` | Pixel-level image crop | `opencv-python` |
| `adaptive_zoom` | Iterative zoom for small text/regions before OCR | `opencv-python` |
| `google_search` / `bing_search` | Web search for entities, products, institutions, public context | `SERPAPI_API_KEY` |
| `amap_poi_search` | China POI/location search | `AMAP_API_KEY` |
| `google_maps_search` | International place search through SerpApi | `SERPAPI_API_KEY` |
| `fetch_webpage` | Fetch and clean webpage text | `trafilatura` |
| `deep_visual_analysis` | Gemini visual specialist for difficult image reasoning | `GOOGLE_KEY_PATH` |

Smoke-test local tools:

```bash
bash scripts/argus/test_local_tools.sh
```

Smoke-test network tools:

```bash
bash scripts/argus/test_local_tools.sh --network
```

Smoke-test OCR:

```bash
GPU=0 ARGUS_TOOL_DEVICE=auto bash scripts/argus/test_local_tools.sh --ocr
```

The local smoke test avoids network calls and avoids loading PaddleOCR unless you pass `--network` or `--ocr`.

## Utility Scripts

```bash
python scripts/argus/count_json_dataset_stats.py --xhs-dir inputs/argus/json/xhs --ins-dir inputs/argus/json/ins
python scripts/argus/summarize_run_outputs.py outputs/argus/run1
```

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
