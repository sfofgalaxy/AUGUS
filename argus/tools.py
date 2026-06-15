"""ARGUS execution tools.

These are plain Python functions exposed to the Investigator as tool calls.
They wrap the ARGUS-local ToolRegistry, which lazy-loads implementations from
`argus/skill_adapters/*/scripts/`.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from argus.path_utils import resolve_local_path

logger = logging.getLogger(__name__)

MAX_RESPONSE_LENGTH = 50_000


def _truncate(text: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    if len(text) > max_length:
        return text[:max_length] + f"\n\n... [Truncated: {len(text)} chars total]"
    return text


def _get_tool(name: str):
    from argus.tool_registry import ToolRegistry
    return ToolRegistry.get(name)


def google_search(query: str, num: int = 5, gl: str = "us") -> str:
    """Search the web using Google via SerpApi.

    Args:
        query: Search query string.
        num: Number of results (1-100).
        gl: Geographic location code (e.g., 'us', 'cn').

    Returns:
        JSON search results.
    """
    try:
        searcher = _get_tool("google_search")
        if searcher is None:
            return "Error: Google search not available. Check SERPAPI_API_KEY."
        result = searcher.search_text(query, num=num)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        return f"Error in Google search: {e}"


def bing_search(query: str, num: int = 5, cc: str = "us") -> str:
    """Search the web using Bing via SerpApi.

    Args:
        query: Search query string.
        num: Number of results (1-50).
        cc: Country code (e.g., 'us', 'cn').

    Returns:
        JSON search results.
    """
    try:
        searcher = _get_tool("bing_search")
        if searcher is None:
            return "Error: Bing search not available. Check SERPAPI_API_KEY."
        result = searcher.search_text(query, num=num, cc=cc)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        return f"Error in Bing search: {e}"


def amap_poi_search(keyword: str, city: Optional[str] = None) -> str:
    """Search for places in China using Amap (Gaode Maps).

    Args:
        keyword: Search keyword (e.g., business name, address).
        city: Optional Chinese city name to restrict search.

    Returns:
        JSON POI results.
    """
    try:
        searcher = _get_tool("amap")
        if searcher is None:
            return "Error: Amap search not available. Check AMAP_API_KEY."
        result = searcher.search_poi(keyword=keyword, city=city)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        return f"Error in Amap search: {e}"


def google_maps_search(query: str, ll: Optional[str] = None) -> str:
    """Search for places using Google Maps.

    Args:
        query: Search query (e.g., business name, landmark).
        ll: Optional location filter as '@lat,lng,zoom' (e.g., '@35.67,139.65,15z').

    Returns:
        JSON place results.
    """
    try:
        searcher = _get_tool("gmaps")
        if searcher is None:
            return "Error: Google Maps search not available."
        result = searcher.search_place(query=query, ll=ll)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        return f"Error in Google Maps search: {e}"


def run_ocr(image_path: str, task: str = "ocr") -> str:
    """Extract text from an image using PaddleOCR-VL-1.5.

    Args:
        image_path: Path to the image file on disk.
        task: Recognition task: 'ocr', 'table', 'formula', 'chart', 'spotting', or 'seal'.

    Returns:
        Recognized text.
    """
    try:
        ocr = _get_tool("ocr")
        if ocr is None:
            return "Error: OCR not available."
        result = ocr.run_ocr(image_path=resolve_local_path(image_path), task=task)
        return _truncate(result)
    except Exception as e:
        return f"Error in OCR: {e}"


def crop_image(image_path: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """Crop a rectangular region from an image.

    Args:
        image_path: Path to the image file on disk.
        x1: Left x coordinate (pixels).
        y1: Top y coordinate (pixels).
        x2: Right x coordinate (pixels).
        y2: Bottom y coordinate (pixels).

    Returns:
        Path to the saved cropped image.
    """
    try:
        cropper = _get_tool("cropper")
        if cropper is None:
            return "Error: Image cropper not available."
        return cropper.crop(resolve_local_path(image_path), [x1, y1, x2, y2])
    except Exception as e:
        return f"Error in image cropping: {e}"


def adaptive_zoom(image_path: str, bbox: str, ratio: float = 0.16) -> str:
    """Zoom into a region of an image using adaptive zoom.

    Args:
        image_path: Path to the image file on disk.
        bbox: Bounding box as JSON array [x1, y1, x2, y2].
        ratio: Area ratio for the crop.

    Returns:
        JSON list of cropped image paths.
    """
    try:
        zoomer = _get_tool("adaptive_zoom")
        if zoomer is None:
            return "Error: Adaptive zoom not available."
        bbox_list = json.loads(bbox)
        result = zoomer.run(resolve_local_path(image_path), [bbox_list], ratios=ratio)
        return _truncate(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as e:
        return f"Error in adaptive zoom: {e}"


def fetch_webpage(url: str, max_retries: int = 3) -> str:
    """Fetch and extract clean text content from a web page.

    Args:
        url: The URL to fetch.
        max_retries: Maximum number of retry attempts.

    Returns:
        JSON with status, title, and extracted text content.
    """
    import time

    fetcher = _get_tool("webpage_fetcher")
    if fetcher is None:
        return "Error: Webpage fetcher not available."

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fetcher.fetch(url=url)
            return _truncate(json.dumps(result, ensure_ascii=False, default=str))
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning("fetch_webpage attempt %d/%d failed: %s", attempt, max_retries, e)
                time.sleep(1 * attempt)
            else:
                logger.error("fetch_webpage failed after %d attempts: %s", max_retries, e)
    return f"Error in webpage fetch after {max_retries} attempts: {last_error}"


def get_all_execution_tools() -> list:
    """Return all execution tools as plain functions for OpenAI tool calling."""
    return [
        google_search,
        bing_search,
        amap_poi_search,
        google_maps_search,
        run_ocr,
        crop_image,
        adaptive_zoom,
        fetch_webpage,
    ]
