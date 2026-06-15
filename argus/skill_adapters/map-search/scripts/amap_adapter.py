"""
Amap (Gaode Maps) POI search adapter.

Provides programmatic access to Amap (Gaode) Web Service API for Point of
Interest (POI) keyword search within China. Supports city-level filtering,
pagination, and returns structured POI data including names, addresses,
coordinates, and business details.

Usage as library:
    from amap_adapter import AmapAdapter

    adapter = AmapAdapter(api_key="your_amap_key")
    result = adapter.search_poi("星巴克", city="北京", page_size=10)
    for poi in result.get("pois", []):
        print(poi["name"], poi["address"])

Usage as CLI:
    python amap_adapter.py "星巴克" --city 北京 --page-size 10
    python amap_adapter.py "隐私政策" --city 上海 --no-citylimit
    AMAP_API_KEY=xxx python amap_adapter.py "数据中心" --city 深圳
"""

import argparse
import json
import os
import sys

import requests


class AmapAdapter:
    """Amap (Gaode) Web Service API adapter for POI search in China.

    Attributes:
        api_key: Amap Web Service API key.
        base_url: Amap POI text search endpoint.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("AMAP_API_KEY", "")
        self.base_url = "https://restapi.amap.com/v5/place/text"

    def search_poi(
        self,
        keyword: str,
        city: str | None = None,
        page_size: int = 20,
        page_num: int = 1,
        citylimit: bool = True,
    ) -> dict:
        """
        POI keyword search.

        Args:
            keyword: Search keyword (e.g. a place name or category).
            city: City name or city code to narrow the search scope.
            page_size: Number of results per page (max 25).
            page_num: Page number for pagination (1-based).
            citylimit: If True, restrict results to the specified city.

        Returns:
            dict: Amap API response. On success, contains 'pois' list
                  with name, address, location, etc. On failure, contains
                  'status', 'reason', and optionally 'infocode'.
        """
        if not self.api_key:
            return {"status": "failed", "reason": "Missing API Key"}

        params = {
            "key": self.api_key,
            "keywords": keyword,
            "city": city,
            "citylimit": citylimit,
            "page_size": page_size,
            "page_num": page_num,
            "output": "json",
            "extensions": "all",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "1":
                return result
            return {
                "status": "failed",
                "reason": result.get("info", "Unknown error"),
                "infocode": result.get("infocode"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    """CLI entry point for Amap POI search."""
    parser = argparse.ArgumentParser(
        description="Amap (Gaode Maps) POI Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python amap_adapter.py '星巴克' --city 北京 --page-size 10",
    )
    parser.add_argument("keyword", help="Search keyword (place name or category)")
    parser.add_argument("--city", default=None,
                        help="City name or code to narrow search scope")
    parser.add_argument("--page-size", type=int, default=20,
                        help="Results per page, max 25 (default: 20)")
    parser.add_argument("--page-num", type=int, default=1,
                        help="Page number for pagination (default: 1)")
    parser.add_argument("--no-citylimit", action="store_true",
                        help="Do not restrict results to specified city")
    parser.add_argument("--api-key", default=None,
                        help="Amap API key (default: AMAP_API_KEY env var)")

    args = parser.parse_args()

    adapter = AmapAdapter(api_key=args.api_key)
    result = adapter.search_poi(
        args.keyword,
        city=args.city,
        page_size=args.page_size,
        page_num=args.page_num,
        citylimit=not args.no_citylimit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
