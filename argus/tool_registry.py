"""ARGUS-local lazy tool registry.

This is copied from the legacy registry shape but deliberately loads only
ARGUS-local adapters from `argus/skill_adapters`.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from argus.config import load_env, project_root

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Lazy-loading singleton for skill-backed execution tools."""

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, Any] = {}
    _config: Dict[str, Any] = {}
    _skills_dir: Path | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
            cls._config = {}
        return cls._instance

    @classmethod
    def configure(cls, config: Dict[str, Any]) -> "ToolRegistry":
        instance = cls()
        cls._config.update(config)
        skills_dir = config.get("skills_dir")
        if skills_dir:
            cls._skills_dir = Path(skills_dir)
        return instance

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls._tools:
            return cls._tools[name]
        tool = cls._create_tool(name)
        if tool is not None:
            cls._tools[name] = tool
        return tool

    @classmethod
    def cleanup(cls) -> None:
        tool_names = list(cls._tools.keys())
        cls._tools.clear()
        cls._config.clear()
        if tool_names:
            logger.info("ARGUS ToolRegistry cleaned up, released: %s", tool_names)

    @classmethod
    def _import_from_skill(cls, skill_name: str, module_name: str):
        module_path = cls._resolve_adapters_dir() / skill_name / "scripts" / f"{module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"ARGUS tool adapter not found: {module_path}")
        safe_skill_name = skill_name.replace("-", "_")
        spec = importlib.util.spec_from_file_location(
            f"argus_skill_adapters.{safe_skill_name}.scripts.{module_name}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import skill script: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _resolve_adapters_dir(cls) -> Path:
        if cls._skills_dir is not None:
            return cls._skills_dir
        configured = cls._config.get("skills_dir")
        if configured:
            cls._skills_dir = Path(configured)
        else:
            cls._skills_dir = Path(__file__).resolve().parent / "skill_adapters"
        return cls._skills_dir

    @classmethod
    def _create_tool(cls, name: str) -> Any:
        config = cls._config

        if name == "google_search":
            mod = cls._import_from_skill("web-search", "google_search")
            return mod.GoogleSearchAdapter(api_key=config.get("serpapi_api_key"))
        if name == "bing_search":
            mod = cls._import_from_skill("web-search", "bing_search")
            return mod.BingSearchAdapter(api_key=config.get("serpapi_api_key"))

        if name == "amap":
            mod = cls._import_from_skill("map-search", "amap_adapter")
            return mod.AmapAdapter(api_key=config.get("amap_api_key"))
        if name == "gmaps":
            mod = cls._import_from_skill("map-search", "gmaps_adapter")
            return mod.GoogleMapsAdapter(api_key=config.get("serpapi_api_key"))

        if name == "ocr":
            mod = cls._import_from_skill("ocr-vision", "ocr_adapter")
            return mod.OCRAdapter(device=config.get("device"))

        if name == "cropper":
            mod = cls._import_from_skill("image-cropper", "cropper")
            return mod.ImageCropper(temp_dir=config.get("image_cropper_temp_dir"))
        if name == "adaptive_zoom":
            mod = cls._import_from_skill("image-cropper", "adaptive_zoom")
            cropper = cls.get("cropper")
            return mod.AdaptiveZoomIn(cropper)

        if name == "webpage_fetcher":
            mod = cls._import_from_skill("webpage-fetcher", "webpage_fetcher")
            return mod.WebpageFetcher()

        logger.warning("Unknown ARGUS tool: %s", name)
        return None


def init_tool_registry() -> None:
    """Configure ARGUS tools from `.env` / process environment."""
    load_env()
    root = project_root()
    os.environ.setdefault("PROJECT_DIR", str(root))
    ToolRegistry.configure({
        "device": os.environ.get("ARGUS_TOOL_DEVICE", "cpu"),
        "serpapi_api_key": os.environ.get("SERPAPI_API_KEY", ""),
        "amap_api_key": os.environ.get("AMAP_API_KEY", ""),
        "image_cropper_temp_dir": str(root / "temp_crops"),
        "skills_dir": str(Path(__file__).resolve().parent / "skill_adapters"),
    })
