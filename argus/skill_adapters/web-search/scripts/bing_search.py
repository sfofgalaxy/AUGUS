"""
Bing Web Search adapter via SerpApi.

Provides programmatic access to Bing web search results through the SerpApi
service. Useful as a fallback or complement to Google search, with support
for pagination and country-specific targeting.

Usage as library:
    from bing_search import BingSearchAdapter

    adapter = BingSearchAdapter(api_key="your_serpapi_key")
    result = adapter.search_text("privacy policy analysis", num=10)
    for item in result.get("organic_results", []):
        print(item["title"], item["link"])

Usage as CLI:
    python bing_search.py "privacy policy analysis" --num 10 --cc us
    python bing_search.py "data breach notification" --num 5 --first 11
    SERPAPI_API_KEY=xxx python bing_search.py "cookie consent banner"
"""

import argparse
import json
import os
import sys

import requests


class BingSearchAdapter:
    """SerpApi Bing Web Search adapter.

    Attributes:
        api_key: SerpApi API key.
        base_url: SerpApi endpoint URL.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self.base_url = "https://serpapi.com/search"

    def search_text(self, query: str, num: int = 5, first: int = 1, cc: str = "cn") -> dict:
        """
        Bing Web Search.

        Args:
            query: Search keywords.
            num: Number of results (1-50).
            first: Starting position for pagination (1-based).
            cc: Country code, e.g. "cn", "gb".

        Returns:
            dict with keys:
                status: "success" | "error" | "failed"
                raw: Full SerpApi response (when available)
                organic_results: List of search results (when successful)
                message / reason: Error description (when failed)
        """
        if not self.api_key:
            return {"status": "failed", "reason": "Missing SerpApi Key"}

        num = max(1, min(int(num), 50))
        first = max(1, int(first))

        params = {
            "engine": "bing",
            "q": query,
            "api_key": self.api_key,
            "first": first,
        }
        params["cc"] = 'cn'

        try:
            resp = requests.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                return {"status": "error", "message": data["error"], "raw": data}

            meta = data.get("search_metadata", {})
            if meta.get("status") and meta.get("status").lower() != "success":
                return {"status": meta.get("status"), "raw": data}

            organic = data.get("organic_results", [])[:num]
            return {"status": "success", "raw": data, "organic_results": organic}

        except requests.RequestException as e:
            return {"status": "error", "message": str(e)}


def main():
    """CLI entry point for Bing web search."""
    parser = argparse.ArgumentParser(
        description="Bing Web Search via SerpApi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python bing_search.py 'privacy policy' --num 10 --cc gb",
    )
    parser.add_argument("query", help="Search keywords")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of results, 1-50 (default: 5)")
    parser.add_argument("--first", type=int, default=1,
                        help="Starting position for pagination (default: 1)")
    parser.add_argument("--cc", default="cn",
                        help="Country code (default: cn)")
    parser.add_argument("--api-key", default=None,
                        help="SerpApi key (default: SERPAPI_API_KEY env var)")

    args = parser.parse_args()

    adapter = BingSearchAdapter(api_key=args.api_key)
    result = adapter.search_text(args.query, num=args.num, first=args.first, cc=args.cc)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
