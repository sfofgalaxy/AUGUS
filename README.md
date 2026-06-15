# AUGUS

AUGUS is the open-source code for **From Posts to Profiles: User-Level Privacy Leakage on Social Media**.

Paper: [https://arxiv.org/abs/2606.06784](https://arxiv.org/abs/2606.06784)

The repo contains:

- `argus/`: AUGUS pipeline, CPEG, hypothesis state, routing, verifier, LLM/tool adapters.
- `baselines/`: text-only, post-wise VLM, single-agent, self-disclosure, HolmesEye-adapted baselines.
- `scripts/argus/run_json_argus.sh`: run AUGUS.
- `scripts/argus/run_json_baseline.sh`: run baselines.
- `argus/skill_adapters/`: bundled tools. Tools are configurable and can be replaced by setting `ARGUS_SKILLS_DIR`.

No private data, API keys, generated outputs, or local server paths are included.

## Install

```bash
git clone git@github.com:sfofgalaxy/AUGUS.git
cd AUGUS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional local OCR:

```bash
pip install -r requirements-ocr.txt
```

## Configure

Edit `.env`. Set only the providers/tools you use.

Backbone:

- `ARGUS_ROUTER_BASE_URL`
- `ARGUS_ROUTER_API_KEY`
- `ARGUS_ROUTER_MODEL`
- `ARGUS_INVESTIGATOR_MODEL`
- `ARGUS_VERIFIER_MODEL`
- `ARGUS_NARRATIVE_MODEL`

Qwen / DashScope:

- `DASHSCOPE_API_KEY`
- `ARGUS_QWEN_TEXT_MODEL`
- `ARGUS_QWEN_VL_MODEL`
- `ARGUS_QWEN_PERCEPTION_MAX_SIDE`
- `ARGUS_QWEN_IMAGE_MAX_BYTES`
- `ARGUS_QWEN_PAYLOAD_MAX_CHARS`

Gemini / Vertex:

- `GOOGLE_KEY_PATH`
- `GOOGLE_CLOUD_LOCATION`
- `ARGUS_GEMINI_MODEL`

Tools:

- `ARGUS_SKILLS_DIR`
- `ARGUS_TOOL_DEVICE`
- `ARGUS_IMAGE_CROPPER_TEMP_DIR`
- `SERPAPI_API_KEY`
- `AMAP_API_KEY`

Runtime:

- `ARGUS_CACHE_DIR`
- `ARGUS_DISABLE_CACHE`
- `ARGUS_CONTINUE_ON_ERROR`
- `ARGUS_LLM_MAX_RETRIES`
- `ARGUS_GEMINI_MAX_RETRIES`
- `ARGUS_LLM_RETRY_BASE_INTERVAL`

## Input

Prepare one JSON file per user, or a directory of user JSON files.

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
      "metadata": {}
    }
  ]
}
```

`media_files` supports absolute paths, JSON-relative paths, repo-relative paths, HTTP(S) URLs, or image data URLs.

## Run AUGUS

```bash
INPUT=examples/sample_user.json \
OUTPUT_DIR=outputs/argus/sample \
bash scripts/argus/run_json_argus.sh
```

Optional:

```bash
GPU=0 INPUT=inputs/users OUTPUT_DIR=outputs/argus/run1 bash scripts/argus/run_json_argus.sh
LIMIT=5 INPUT=inputs/users OUTPUT_DIR=outputs/argus/debug bash scripts/argus/run_json_argus.sh
```

## Run Baselines

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

Baseline names:

- `text_only`
- `single_post`
- `single_agent`
- `self_disclosure_detector`
- `holmeseye_adapted`

Model override:

```bash
MODEL_PROVIDER=qwen MODEL=qwen3.7-max BASELINE=text_only INPUT=inputs/users OUTPUT_DIR=outputs/baselines/qwen_text \
bash scripts/argus/run_json_baseline.sh
```

## Tool Check

```bash
bash scripts/argus/test_local_tools.sh
bash scripts/argus/test_local_tools.sh --network
GPU=0 ARGUS_TOOL_DEVICE=auto bash scripts/argus/test_local_tools.sh --ocr
```

## Citation

```bibtex
@misc{augus2026posts,
  title = {From Posts to Profiles: User-Level Privacy Leakage on Social Media},
  year = {2026},
  eprint = {2606.06784},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2606.06784}
}
```
