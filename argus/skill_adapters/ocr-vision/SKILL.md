---
name: ocr-vision
description: "Extract text from images using PaddleOCR-VL-1.5. USE when: you need to re-OCR a cropped/zoomed region for better text extraction, or when initial perception OCR missed details. Note: perception node already runs OCR on all images — use this for targeted re-extraction on specific regions."
tools:
  - run_ocr
---

# OCR Vision

Extract text from images using PaddleOCR-VL-1.5 (HuggingFace Transformers). The model supports multiple recognition tasks via a `task` parameter — choose the right one for the content type.

## When to Use

- Reading text visible in photos (signs, labels, receipts, documents)
- Extracting text from screenshots or social media posts
- Recognizing tables → HTML structure
- Recognizing mathematical formulas → LaTeX
- Reading seal/stamp text
- Processing cropped image regions for detailed text extraction

## Available Tasks

| Task | Prompt | Output | Best For |
|------|--------|--------|----------|
| `ocr` | General OCR | Plain text | Default — screenshots, photos, documents |
| `table` | Table recognition | HTML | Spreadsheet-like content, data tables |
| `formula` | Formula recognition | LaTeX | Math equations, scientific notation |
| `chart` | Chart recognition | Text | Diagrams, charts with labels |
| `spotting` | Fine-grained spotting | Text | Small/dense text (auto-upscales) |
| `seal` | Seal recognition | Text | Stamps, seals, circular text |

## How to Use

### Basic OCR (default)

```
run_ocr(image_path="/path/to/image.jpg")
```

Returns plain text extracted from the image.

### Task-Specific Recognition

```
run_ocr(image_path="/path/to/table.png", task="table")
run_ocr(image_path="/path/to/equation.png", task="formula")
run_ocr(image_path="/path/to/stamp.jpg", task="seal")
```

### Multi-Step Workflow

**Step 1: Assess image quality**
- Is the text large enough? If not, use `image-cropper` skill first to crop/zoom
- Does the image contain a table, formula, or seal? Choose the right `task`

**Step 2: Run OCR with the appropriate task**
- Most images → `task="ocr"` (default)
- Tables/spreadsheets → `task="table"` (returns HTML)
- Math equations → `task="formula"` (returns LaTeX)
- Small/dense text → `task="spotting"` (auto-upscales small images)

**Step 3: Handle poor results**
1. Use `image-cropper` to zoom into the specific region
2. Re-run OCR on the cropped image
3. Try `task="spotting"` for hard-to-read text

**Step 4: Extract privacy-relevant information**
Look for: names, addresses, phone numbers, financial info, location clues, dates.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/ocr_adapter.py` | OCRAdapter — PaddleOCR-VL-1.5 via HuggingFace Transformers |

### Script CLI Usage

```bash
# Default OCR
python scripts/ocr_adapter.py image.png

# Table recognition
python scripts/ocr_adapter.py table.png --task table

# Formula recognition
python scripts/ocr_adapter.py equation.png --task formula

# Seal recognition
python scripts/ocr_adapter.py stamp.jpg --task seal
```

## Resources (load on-demand only)

Consult references via `read_skill_file("ocr-vision", "references/paddleocr_guide.md")` ONLY when:
- OCR results are poor quality and you need troubleshooting strategies
- You encounter memory errors and need resolution steps
- You need details about task-specific behavior or max_pixels settings

Reference files:
- `references/paddleocr_guide.md` - Model details, task descriptions, and troubleshooting

## Notes

- Model loaded lazily on first use (~2-4GB memory, supports CPU and CUDA)
- Supports multilingual text: Chinese, English, Japanese, Korean, etc.
- Image must exist as a local file (no URL support)
- For small/unclear text, crop and zoom first using `image-cropper`
- `spotting` task auto-upscales images below 1500px
