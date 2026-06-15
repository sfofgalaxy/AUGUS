"""
OCR adapter using PaddleOCR-VL-1.5 via HuggingFace Transformers.

Supports multiple recognition tasks via prompt:
  - ocr       : General text extraction (default)
  - table     : Table structure recognition (returns HTML)
  - chart     : Chart/diagram text recognition
  - formula   : Mathematical formula recognition (returns LaTeX)
  - spotting  : Fine-grained text spotting (upscales small images)
  - seal      : Seal/stamp text recognition

Usage as library:
    from ocr_adapter import OCRAdapter

    ocr = OCRAdapter()
    text = ocr.run_ocr("screenshot.png")                # plain OCR
    html = ocr.run_ocr("table.png", task="table")       # table → HTML
    latex = ocr.run_ocr("equation.png", task="formula")  # formula → LaTeX

Usage as CLI:
    python ocr_adapter.py image.png
    python ocr_adapter.py image.png --task table
    python ocr_adapter.py image.png --task formula
"""

import argparse
import os
import sys
from typing import Optional

from argus.logging_utils import argus_log


# Task → prompt mapping (from official PaddleOCR-VL-1.5 documentation)
TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

VALID_TASKS = list(TASK_PROMPTS.keys())

# Spotting task upscale threshold (from official code)
SPOTTING_UPSCALE_THRESHOLD = 1500


class OCRAdapter:
    """
    OCR adapter using PaddleOCR-VL-1.5 (HuggingFace Transformers).

    The model is loaded lazily on first use. Supports CPU and CUDA.

    Attributes:
        model: AutoModelForImageTextToText instance (lazy-loaded).
        processor: AutoProcessor instance (lazy-loaded).
        device: 'cuda' or 'cpu'.
    """

    MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

    def __init__(self, device: Optional[str] = None) -> None:
        self._model = None
        self._processor = None
        if device and device != "auto":
            self._device = device
        else:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

    def _ensure_loaded(self):
        """Lazy-load model and processor on first use."""
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        argus_log(f"Loading PaddleOCR-VL-1.5 on {self._device} ...")
        try:
            dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            if self._model is None:
                self._model = (
                    AutoModelForImageTextToText
                    .from_pretrained(self.MODEL_ID, dtype=dtype)
                    .to(self._device)
                    .eval()
                )
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PaddleOCR-VL-1.5: {e}\n"
                f"  Model loaded: {self._model is not None}\n"
                f"  Processor loaded: {self._processor is not None}\n"
                f"  Device: {self._device}"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_ocr(self, image_path: str, task: str = "ocr") -> str:
        """
        Run OCR on an image with the specified task.

        Args:
            image_path: Path to the image file.
            task: Recognition task — one of:
                  'ocr' (default), 'table', 'chart', 'formula',
                  'spotting', 'seal'.

        Returns:
            Recognized text (plain text, HTML for tables, LaTeX for formulas).

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If task is not recognized.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        if task not in TASK_PROMPTS:
            raise ValueError(
                f"Unknown task '{task}'. Valid tasks: {VALID_TASKS}"
            )

        self._ensure_loaded()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # Spotting task: upscale small images (official recommendation)
        if (
            task == "spotting"
            and orig_w < SPOTTING_UPSCALE_THRESHOLD
            and orig_h < SPOTTING_UPSCALE_THRESHOLD
        ):
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize((orig_w * 2, orig_h * 2), resample)

        # Max pixels: spotting uses larger context, others use default ~1M
        if task == "spotting":
            max_pixels = 2048 * 28 * 28
        else:
            max_pixels = 1280 * 28 * 28

        # Table/chart may produce long HTML; others stay within 512 (official default)
        max_tokens = 4096 if task in ("table", "chart") else 512

        # Build chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TASK_PROMPTS[task]},
                ],
            }
        ]

        # Tokenize
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self._processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(self._model.device)

        # Generate
        import torch
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                repetition_penalty=1.2,
            )

        # Decode (skip input tokens and trailing EOS)
        result = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:-1]
        )

        return result.strip()

    # Backward-compatible aliases used by ToolRegistry / execution_tools
    def run_ocr_structured(self, image_path: str) -> dict:
        """
        Backward-compatible structured interface.

        Returns a dict with 'text' key containing OCR result.
        (The old PaddleOCR block-based layout is no longer used.)
        """
        text = self.run_ocr(image_path, task="ocr")
        return {
            "image_path": image_path,
            "text": text,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OCR using PaddleOCR-VL-1.5 (HuggingFace Transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ocr_adapter.py screenshot.png\n"
            "  python ocr_adapter.py receipt.jpg --task ocr\n"
            "  python ocr_adapter.py table.png --task table\n"
            "  python ocr_adapter.py equation.png --task formula\n"
            "  python ocr_adapter.py stamp.jpg --task seal\n"
        ),
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--task",
        default="ocr",
        choices=VALID_TASKS,
        help="Recognition task (default: ocr)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    ocr = OCRAdapter()
    result = ocr.run_ocr(args.image_path, task=args.task)
    print(result)


if __name__ == "__main__":
    main()
