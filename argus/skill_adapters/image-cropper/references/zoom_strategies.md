# Zoom Strategies Guide

## Adaptive Zoom Algorithm

The adaptive zoom calculates a crop window centered on the target region:

1. **Compute side factor**: `side = sqrt(area_ratio)`
2. **Calculate target dimensions**: `target_w = image_w * side`, `target_h = image_h * side`
3. **Center on ROI**: Place crop window centered on bbox midpoint
4. **Boundary shift**: If crop exceeds image edge, shift (don't shrink) to maintain area

### Example

Image: 1920 x 1080, ratio: 0.16
- side = sqrt(0.16) = 0.4
- target: 768 x 432 pixels
- Crop maintains exactly 16% of original image area

## Ratio Selection Guide

### By Content Type

| Content | Recommended Ratio | Reasoning |
|---------|-------------------|-----------|
| Street signs | 0.05 - 0.10 | Small text needs high zoom |
| Product labels | 0.08 - 0.12 | Medium detail |
| Documents | 0.15 - 0.25 | Larger text blocks |
| Storefront | 0.20 - 0.30 | Building-level detail |
| Scene overview | 0.40 - 0.60 | Half-scene context |

### By OCR Quality Issue

| Problem | Solution |
|---------|----------|
| Text too small to read | Use ratio 0.05, re-run OCR |
| Partial text recognized | Use ratio 0.10-0.15 for more context |
| Noisy background | Tight crop (0.05) to isolate text |
| Rotated text | Crop region, then rotate externally |

## Multi-ROI Processing

When multiple regions need attention:

```python
bboxes = [
    [0.1, 0.1, 0.3, 0.2],   # Top-left sign
    [0.5, 0.3, 0.7, 0.5],   # Center label
    [0.8, 0.7, 0.95, 0.9],  # Bottom-right text
]
ratios = [0.05, 0.10, 0.08]  # Different ratios per region

crop_paths = zoomer.run(image_path, bboxes, ratios, max_crops=5)
```

### Per-ROI Ratio Assignment

Assign different ratios based on each region's content:
- Smaller text → smaller ratio (more zoom)
- Larger text → larger ratio (more context)

## Iterative Zooming

For extremely small details:

1. First crop at ratio 0.16 (medium zoom)
2. Run OCR on the crop
3. If still unreadable, crop the crop at ratio 0.16 again
4. This gives effective zoom of 0.16 * 0.16 = 0.0256 (~2.5% of original)

Each iteration provides ~2.5x magnification.
