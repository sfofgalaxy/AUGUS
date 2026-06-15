"""
Image cropper — atomic tool for bounding-box-based image cropping.

Physically crops a rectangular region from an image and saves the result to a
temp directory. Includes automatic boundary clamping to prevent out-of-bounds
coordinates from causing errors, which is important when an agent supplies
imprecise bounding boxes.

Usage as library:
    from cropper import ImageCropper

    cropper = ImageCropper(temp_dir="./temp_crops")
    crop_path = cropper.crop("screenshot.png", [100, 200, 400, 500])
    print(f"Cropped image saved to: {crop_path}")

Usage as CLI:
    python cropper.py screenshot.png 100 200 400 500
    python cropper.py photo.jpg 0 0 300 300 --temp-dir /tmp/crops
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2


PROJECT_DIR_ENV = "PROJECT_DIR"


def get_project_dir() -> Path:
    """Load the project root from the PROJECT_DIR environment variable."""
    raw_project_dir = os.getenv(PROJECT_DIR_ENV, "").strip()
    if not raw_project_dir:
        raise ValueError("PROJECT_DIR is not set.")
    return Path(raw_project_dir).expanduser().resolve()


def resolve_temp_dir(temp_dir: Optional[str]) -> Path:
    """Resolve crop output directory to an absolute project-local path."""
    project_dir = get_project_dir()
    if temp_dir:
        resolved = Path(temp_dir).expanduser()
        if not resolved.is_absolute():
            resolved = project_dir / resolved
        return resolved.resolve()
    return (project_dir / "temp_crops").resolve()


class ImageCropper:
    """
    Atomic image cropping tool.

    Crops a rectangular region specified by pixel-coordinate bounding box
    [x1, y1, x2, y2] and saves the result as a JPEG file.

    Attributes:
        temp_dir: Directory for saving cropped image files.
    """

    def __init__(self, temp_dir: Optional[str] = None) -> None:
        self.temp_dir = resolve_temp_dir(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def crop(self, image_path: str, bbox: list[int]) -> str:
        """
        Crop an image region and return the saved file path.

        Args:
            image_path: Path to the source image.
            bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            str: Path to the saved cropped image.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If the image cannot be read or bbox is invalid.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image content: {image_path}")

        h, w, _ = img.shape
        x1, y1, x2, y2 = map(int, bbox)

        # Boundary clamping: prevent agent-supplied coordinates from overflowing
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop region: {x1, y1, x2, y2}")

        crop_img = img[y1:y2, x1:x2]

        # Unique ID to prevent filename collisions
        filename = f"crop_{uuid.uuid4().hex[:8]}.jpg"
        save_path = self.temp_dir / filename
        cv2.imwrite(str(save_path), crop_img)

        return str(save_path)


def main():
    """CLI entry point for image cropping."""
    parser = argparse.ArgumentParser(
        description="Crop a rectangular region from an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python cropper.py screenshot.png 100 200 400 500",
    )
    parser.add_argument("image_path", help="Path to the source image")
    parser.add_argument("x1", type=int, help="Left x coordinate")
    parser.add_argument("y1", type=int, help="Top y coordinate")
    parser.add_argument("x2", type=int, help="Right x coordinate")
    parser.add_argument("y2", type=int, help="Bottom y coordinate")
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

    cropper = ImageCropper(temp_dir=args.temp_dir)
    crop_path = cropper.crop(args.image_path, [args.x1, args.y1, args.x2, args.y2])
    print(json.dumps({"status": "success", "crop_path": crop_path}, indent=2))


if __name__ == "__main__":
    main()
