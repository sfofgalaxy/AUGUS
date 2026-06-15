"""
Google Web Search adapter via SerpApi.

Provides programmatic access to Google web search results through the SerpApi
service. Supports pagination, geographic targeting, and returns structured
result data including organic results.

Usage as library:
    from google_search import GoogleSearchAdapter

    adapter = GoogleSearchAdapter(api_key="your_serpapi_key")
    result = adapter.search_text("privacy policy analysis", num=10)
    for item in result.get("organic_results", []):
        print(item["title"], item["link"])

Usage as CLI:
    python google_search.py "privacy policy analysis" --num 10
    python google_search.py "data breach notification" --num 5 --start 10
    SERPAPI_API_KEY=xxx python google_search.py "GDPR compliance"
"""

import argparse
import json
import os
import sys

import requests


class GoogleSearchAdapter:
    """SerpApi Google Web Search adapter.

    Attributes:
        api_key: SerpApi API key.
        base_url: SerpApi endpoint URL.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self.base_url = "https://serpapi.com/search"

    def search_text(self, query: str, num: int = 5, start: int = 0) -> dict:
        """
        Google Web Search.

        Args:
            query: Search keywords.
            num: Number of results (1-100).
            start: Result offset for pagination.

        Returns:
            dict with keys:
                status: "success" | "error" | "failed"
                raw: Full SerpApi response (when available)
                organic_results: List of search results (when successful)
                message / reason: Error description (when failed)
        """
        if not self.api_key:
            return {"status": "failed", "reason": "Missing SerpApi Key"}

        num = max(1, min(int(num), 100))
        start = max(0, int(start))

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "start": start,
        }

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
    """CLI entry point for Google web search."""
    parser = argparse.ArgumentParser(
        description="Google Web Search via SerpApi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python google_search.py 'privacy policy' --num 10",
    )
    parser.add_argument("query", help="Search keywords")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of results, 1-100 (default: 5)")
    parser.add_argument("--start", type=int, default=0,
                        help="Result offset for pagination (default: 0)")
    parser.add_argument("--api-key", default=None,
                        help="SerpApi key (default: SERPAPI_API_KEY env var)")

    args = parser.parse_args()

    adapter = GoogleSearchAdapter(api_key=args.api_key)
    result = adapter.search_text(args.query, num=args.num, start=args.start)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
