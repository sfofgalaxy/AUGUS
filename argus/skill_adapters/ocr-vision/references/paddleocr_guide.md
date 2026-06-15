# PaddleOCR-VL-1.5 Guide

## Model

- **HuggingFace**: `PaddlePaddle/PaddleOCR-VL-1.5`
- **Backend**: HuggingFace Transformers (`AutoModelForImageTextToText`)
- **Precision**: bfloat16
- **Memory**: ~2-4GB

## Tasks

The model uses a prompt-based interface. Each task has a specific prompt:

| Task | Prompt | Output Format | Notes |
|------|--------|---------------|-------|
| `ocr` | `OCR:` | Plain text | General-purpose text extraction |
| `table` | `Table Recognition:` | HTML `<table>` | Structured table output |
| `formula` | `Formula Recognition:` | LaTeX | Math/scientific formulas |
| `chart` | `Chart Recognition:` | Plain text | Charts and diagrams |
| `spotting` | `Spotting:` | Plain text | Fine-grained, auto-upscales small images |
| `seal` | `Seal Recognition:` | Plain text | Circular/stamp text |

## Max Pixels

Controls image resolution sent to the model:

| Task | max_pixels | Equivalent |
|------|------------|------------|
| `spotting` | `2048 * 28 * 28` = 1,605,632 | Higher resolution for fine detail |
| All others | `1280 * 28 * 28` = 1,003,520 | Standard ~1M pixels |

## Spotting Upscale

For `spotting` task, images smaller than 1500px on both dimensions are upscaled 2x using LANCZOS resampling before processing. This improves detection of small/dense text.

## Supported Languages

Auto-detected, no configuration needed:
- Chinese (Simplified & Traditional)
- English
- Japanese, Korean
- French, German, Spanish
- Arabic
- 70+ other languages

## Performance Tips

### Image Quality

| Quality | Expected Accuracy |
|---------|-------------------|
| High-res scan (300+ DPI) | 95%+ |
| Clear photo (1080p+) | 90%+ |
| Low-res screenshot | 80%+ |
| Blurry/rotated | 60-80% |

### Improving Results

1. **Crop region of interest** before OCR for small text
2. **Use `spotting` task** for dense/small text
3. **Use `table` task** for tabular data (gets HTML structure)
4. **Use `formula` task** for equations (gets LaTeX)

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Check image path exists |
| CUDA out of memory | Use CPU (`OCRAdapter(device="cpu")`) or reduce image size |
| Empty/garbled result | Try different task, or crop+zoom the region first |
| Slow first run | Model download (~2GB) happens on first use, cached after |
| `torch.bfloat16` not supported | Older GPUs — update PyTorch or use CPU |
