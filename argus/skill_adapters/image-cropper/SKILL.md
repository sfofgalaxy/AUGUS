---
name: image-cropper
description: "Crop and zoom into image regions for detailed analysis. USE when: OCR returned empty/partial results on an image that visually contains text (signs, documents, receipts, screens), or when you need to isolate a specific region (logo, sign, document) for re-OCR or visual follow-up."
tools:
  - crop_image
  - adaptive_zoom
---

# Image Cropper

Crop rectangular regions and perform adaptive zoom on images for detailed visual analysis.

## When to Use

- Text in an image is too small to read with OCR
- Need to focus on a specific object or region in a photo
- Extracting details from a larger scene (signs, labels, screens)
- Preparing image regions for OCR or visual follow-up
- Iterative zoom-in on areas of interest

## Available Operations

1. **Basic Crop**: Cut a rectangular region by pixel coordinates
2. **Adaptive Zoom**: Intelligent zoom centered on a point with configurable area ratio

## Multi-Step Workflow

### Step 1: Identify Region of Interest

Determine what needs closer inspection:
- Use OCR bounding boxes from a previous `run_ocr` call
- Visual inspection of the image
- Coordinates from image analysis

### Step 2: Choose Crop Method

- **Basic crop** (`crop_image`): When you have exact pixel coordinates [x1, y1, x2, y2]
- **Adaptive zoom** (`adaptive_zoom`): When you want intelligent zoom around a center point

### Step 3: Select Zoom Ratio (for adaptive zoom)

| Ratio | Description | Use Case |
|-------|-------------|----------|
| 0.05 | Very small area | Tiny text, small icons |
| 0.10 | Small area | Small labels, buttons |
| 0.16 | Default | General detail inspection |
| 0.25 | Large area | Posters, larger signs |
| 0.50 | Half image | Wide scene focus |

### Step 4: Execute Crop

```
crop_image(image_path="/data/photo.jpg", x1=100, y1=200, x2=500, y2=400)
adaptive_zoom(image_path="/data/photo.jpg", bbox="[0.3, 0.4, 0.5, 0.6]", ratio=0.10)
```

### Step 5: Process Cropped Result

Use the cropped image with other skills:
- `run_ocr` for text extraction on the zoomed region
- web/map search on text or entities extracted from the crop
- Further zoom if still insufficient detail

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/cropper.py` | ImageCropper - Pixel-level crop with boundary protection and auto-save |
| `scripts/adaptive_zoom.py` | AdaptiveZoomIn - Center-based zoom with dynamic area ratios and boundary shift |

### Script CLI Usage

Scripts can be called directly from the command line:

```bash
# Basic crop with pixel coordinates [x1, y1, x2, y2]
python scripts/cropper.py screenshot.png 100 200 400 500
python scripts/cropper.py photo.jpg 0 0 300 300 --temp-dir /tmp/crops

# Adaptive zoom with JSON bounding boxes
python scripts/adaptive_zoom.py img.png '[[100,200,300,400]]'
python scripts/adaptive_zoom.py img.png '[[0.1,0.2,0.3,0.4]]' --ratio 0.25
python scripts/adaptive_zoom.py img.png '[[50,50,200,200],[300,300,500,500]]' --max-crops 3
```

Output: JSON to stdout with `status` and `crop_path` / `crop_paths` fields.

## Resources (load on-demand only)

Consult references via `read_skill_file("image-cropper", "references/zoom_strategies.md")` ONLY when:
- The ratio table above is insufficient and you need fine-grained ratio selection by content type
- You're doing iterative zoom (crop-of-crop) and need to calculate effective zoom percentages
- You need the exact mathematical formula for the adaptive zoom algorithm

Reference files:
- `references/zoom_strategies.md` - Adaptive zoom algorithm details, ratio selection guide, and iterative zoom patterns

## Examples

### Example 1: Crop a Sign

After identifying a sign at coordinates [100, 200, 500, 400]:
```
crop_image(image_path="/data/street_photo.jpg", x1=100, y1=200, x2=500, y2=400)
```
Returns: Path to cropped image file.

### Example 2: Zoom into Small Text

Using normalized coordinates from OCR bbox:
```
adaptive_zoom(image_path="/data/receipt.jpg", bbox="[0.1, 0.8, 0.4, 0.95]", ratio=0.05)
```

### Example 3: Iterative Zoom

First zoom:
```
adaptive_zoom(image_path="/data/photo.jpg", bbox="[0.3, 0.3, 0.7, 0.7]", ratio=0.16)
```
Then zoom again on the result for even more detail:
```
adaptive_zoom(image_path="/temp_crops/crop_abc123.jpg", bbox="[0.4, 0.4, 0.6, 0.6]", ratio=0.05)
```

## Notes

- Cropped images saved to `./temp_crops/` directory by default
- Boundary protection: coordinates are automatically clamped to image dimensions
- Supports both pixel coordinates and normalized [0-1] coordinates
- Invalid crop regions (zero area) raise an error
- Adaptive zoom preserves the target area ratio using boundary shift strategy
- OpenCV is required for image processing
