"""
Adaptive Zoom — crop image regions with dynamic area ratios.

Given a list of bounding boxes (normalized or pixel coordinates) and optional
area ratios, computes context-aware crop regions centered on each target area.
Uses boundary shift strategy to preserve the desired area ratio even when
targets are near image edges.

The cropper instance is injected via constructor (from ToolRegistry),
so this module has no import dependency on cropper.py.

Usage as library:
    from cropper import ImageCropper
    from adaptive_zoom import AdaptiveZoomIn

    cropper = ImageCropper(temp_dir="./temp_crops")
    zoom = AdaptiveZoomIn(cropper=cropper, default_ratio=0.16)
    crop_paths = zoom.run(
        "screenshot.png",
        bboxes=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        ratios=0.25,
    )
    for path in crop_paths:
        print(f"Zoomed crop: {path}")

Usage as CLI:
    python adaptive_zoom.py screenshot.png "[[100,200,300,400]]"
    python adaptive_zoom.py photo.jpg "[[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]" --ratio 0.25
    python adaptive_zoom.py img.png "[[50,50,200,200]]" --max-crops 3 --temp-dir /tmp/crops
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Union

import cv2


class AdaptiveZoomIn:
    """
    Adaptive zoom into image regions with dynamic area ratios.

    Centers a crop window on each target bounding box, scaling the window
    to cover the specified fraction of the full image area. Handles both
    normalized [0-1] and pixel-coordinate bounding boxes.

    Attributes:
        cropper: An ImageCropper instance (or any object with a
                 ``crop(image_path, bbox)`` method).
        default_ratio: Default area ratio when not specified per-bbox.
    """

    def __init__(self, cropper, default_ratio: float = 0.16) -> None:
        self.cropper = cropper
        self.default_ratio = default_ratio

    def run(self, image_path: str, bboxes: List[List[float]],
            ratios: Optional[Union[float, List[float]]] = None,
            max_crops: int = 5) -> List[str]:
        """
        Adaptive zoom into image regions.

        Args:
            image_path: Path to the image (supports original or previously cropped).
            bboxes: List of bounding boxes, normalized [0-1] or pixel coordinates.
            ratios: Dynamic area ratios. Single float or per-bbox list.
                    Examples: 0.05 (tiny text), 0.16 (default), 0.25 (poster).
            max_crops: Maximum number of crops to produce.

        Returns:
            List of file paths to saved cropped images.

        Raises:
            ValueError: If image cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        h, w, _ = img.shape
        crop_paths: List[str] = []

        # Handle dynamic ratio logic
        if ratios is None:
            current_ratios = [self.default_ratio] * len(bboxes)
        elif isinstance(ratios, float):
            current_ratios = [ratios] * len(bboxes)
        else:
            current_ratios = ratios

        for i, bbox in enumerate(bboxes[:max_crops]):
            x1, y1, x2, y2 = bbox
            ratio = current_ratios[i] if i < len(current_ratios) else self.default_ratio

            # Compute side length scaling (side = sqrt(area_ratio))
            side_factor = math.sqrt(ratio)
            target_w, target_h = w * side_factor, h * side_factor

            # 1. Coordinate conversion and center point calculation
            if all(0.0 <= c <= 1.0 for c in bbox):
                cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
            else:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 2. Compute new coordinates
            nx1, ny1 = int(cx - target_w / 2), int(cy - target_h / 2)
            nx2, ny2 = int(cx + target_w / 2), int(cy + target_h / 2)

            # 3. Boundary shift strategy — ensure area ratio is preserved
            if nx1 < 0:
                nx2 = min(w, nx2 - nx1)
                nx1 = 0
            if ny1 < 0:
                ny2 = min(h, ny2 - ny1)
                ny1 = 0
            if nx2 > w:
                nx1 = max(0, nx1 - (nx2 - w))
                nx2 = w
            if ny2 > h:
                ny1 = max(0, ny1 - (ny2 - h))
                ny2 = h

            crop_paths.append(self.cropper.crop(image_path, [nx1, ny1, nx2, ny2]))

        return crop_paths


def main():
    """CLI entry point for adaptive zoom cropping."""
    parser = argparse.ArgumentParser(
        description="Adaptive zoom — crop image regions with dynamic area ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python adaptive_zoom.py img.png '[[100,200,300,400]]'\n"
               "  python adaptive_zoom.py img.png '[[0.1,0.2,0.3,0.4]]' --ratio 0.25",
    )
    parser.add_argument("image_path", help="Path to the source image")
    parser.add_argument("bboxes", help="JSON array of bounding boxes, e.g. '[[x1,y1,x2,y2],...]'")
    parser.add_argument("--ratio", type=float, default=0.16,
                        help="Area ratio for all bboxes (default: 0.16)")
    parser.add_argument("--max-crops", type=int, default=5,
                        help="Maximum number of crops (default: 5)")
    parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Directory for saving cropped images. "
            "Defaults to PROJECT_DIR/temp_crops"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        bboxes = json.loads(args.bboxes)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON for bboxes: {e}", file=sys.stderr)
        sys.exit(1)

    # Instantiate a cropper for CLI mode
    from cropper import ImageCropper
    cropper = ImageCropper(temp_dir=args.temp_dir)
    zoom = AdaptiveZoomIn(cropper=cropper, default_ratio=args.ratio)
    paths = zoom.run(args.image_path, bboxes, ratios=args.ratio, max_crops=args.max_crops)
    print(json.dumps({"status": "success", "crop_paths": paths}, indent=2))


if __name__ == "__main__":
    main()
