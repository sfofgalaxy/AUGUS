"""Local ARGUS tool smoke tests.

Default mode avoids network calls and avoids loading the PaddleOCR model.
Use flags for heavier checks.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from argus.config import load_env, project_root
from argus.tool_registry import ToolRegistry, init_tool_registry
from argus.tools import get_all_execution_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test ARGUS local tools.")
    parser.add_argument("--network", action="store_true", help="Call search/map/fetch APIs.")
    parser.add_argument("--ocr", action="store_true", help="Load PaddleOCR and run one OCR image.")
    args = parser.parse_args()

    load_env()
    init_tool_registry()

    results = {
        "project_root": str(project_root()),
        "tool_count": len(get_all_execution_tools()),
        "imports": {},
        "local_crop": {},
        "network": {},
        "ocr": {},
    }

    for name in [
        "google_search",
        "bing_search",
        "amap",
        "gmaps",
        "webpage_fetcher",
        "cropper",
        "adaptive_zoom",
    ]:
        try:
            results["imports"][name] = ToolRegistry.get(name).__class__.__name__
        except Exception as exc:
            results["imports"][name] = f"ERROR: {exc}"

    try:
        results["local_crop"] = _test_cropper()
    except Exception as exc:
        results["local_crop"] = {"status": "error", "message": str(exc)}

    if args.network:
        results["network"] = _test_network()
    else:
        results["network"] = {"status": "skipped", "reason": "pass --network to call APIs"}

    if args.ocr:
        results["ocr"] = _test_ocr()
    else:
        results["ocr"] = {"status": "skipped", "reason": "pass --ocr to load PaddleOCR"}

    print(json.dumps(results, ensure_ascii=False, indent=2))


def _test_cropper() -> dict:
    from PIL import Image

    cropper = ToolRegistry.get("cropper")
    zoomer = ToolRegistry.get("adaptive_zoom")
    with tempfile.TemporaryDirectory() as d:
        img_path = Path(d) / "smoke.png"
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path)
        crop_path = cropper.crop(str(img_path), [0, 0, 32, 32])
        zoom_paths = zoomer.run(str(img_path), [[0.25, 0.25, 0.75, 0.75]], ratios=0.25)
        return {
            "status": "success",
            "crop_exists": Path(crop_path).exists(),
            "zoom_count": len(zoom_paths),
        }


def _test_network() -> dict:
    out = {}
    checks = [
        ("google_search", lambda t: t.search_text("OpenAI", num=1)),
        ("bing_search", lambda t: t.search_text("OpenAI", num=1)),
        ("amap", lambda t: t.search_poi(keyword="北京大学", city="北京")),
        ("gmaps", lambda t: t.search_place(query="OpenAI San Francisco")),
        ("webpage_fetcher", lambda t: t.fetch("https://example.com")),
    ]
    for name, fn in checks:
        try:
            result = fn(ToolRegistry.get(name))
            out[name] = {"status": result.get("status", "ok")}
        except Exception as exc:
            out[name] = {"status": "error", "message": str(exc)}
    return out


def _test_ocr() -> dict:
    from PIL import Image, ImageDraw

    ocr = ToolRegistry.get("ocr")
    with tempfile.TemporaryDirectory() as d:
        img_path = Path(d) / "ocr.png"
        image = Image.new("RGB", (320, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((20, 40), "ARGUS TEST 123", fill=(0, 0, 0))
        image.save(img_path)
        text = ocr.run_ocr(str(img_path), task="ocr")
        return {"status": "success", "text_preview": text[:120]}


if __name__ == "__main__":
    main()
