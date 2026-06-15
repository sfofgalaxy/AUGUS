"""
Google Maps Local Results adapter via SerpApi.

Provides programmatic access to Google Maps place search through SerpApi.
Searches for local businesses, landmarks, and points of interest. Supports
geographic targeting via latitude/longitude and pagination.

Usage as library:
    from gmaps_adapter import GoogleMapsAdapter

    adapter = GoogleMapsAdapter(api_key="your_serpapi_key")
    result = adapter.search_place("coffee shop near me", ll="@40.7128,-74.0060,15z")
    for place in result.get("local_results", []):
        print(place["title"], place.get("address"))

Usage as CLI:
    python gmaps_adapter.py "coffee shop" --ll "@40.7128,-74.0060,15z"
    python gmaps_adapter.py "data center" --start 20
    SERPAPI_API_KEY=xxx python gmaps_adapter.py "privacy consultant"
"""

import argparse
import json
import os
import sys

import requests


class GoogleMapsAdapter:
    """SerpApi Google Maps search adapter.

    Attributes:
        api_key: SerpApi API key.
        base_url: SerpApi endpoint URL.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self.base_url = "https://serpapi.com/search"

    def search_place(self, query: str, start: int | None = None, ll: str | None = None) -> dict:
        """
        Google Maps place search.

        Args:
            query: Place name or search keywords.
            start: Starting position for pagination.
            ll: Geographic coordinates in "@lat,lng,zoom" format
                (e.g. "@40.7128,-74.0060,15z").

        Returns:
            dict: Full SerpApi response with local_results, place_results, etc.
                  On error, contains 'status' and 'message' keys.
        """
        if not self.api_key:
            return {"status": "failed", "reason": "Missing API Key"}

        params = {
            "engine": "google_maps",
            "type": "search",
            "q": query,
            "api_key": self.api_key,
        }
        if start is not None:
            params["start"] = start
        if ll:
            params["ll"] = ll

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


def main():
    """CLI entry point for Google Maps place search."""
    parser = argparse.ArgumentParser(
        description="Google Maps Place Search via SerpApi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python gmaps_adapter.py 'coffee shop' --ll '@40.71,-74.00,15z'",
    )
    parser.add_argument("query", help="Place name or search keywords")
    parser.add_argument("--start", type=int, default=None,
                        help="Starting position for pagination")
    parser.add_argument("--ll", default=None,
                        help="Lat/lng coordinates in '@lat,lng,zoom' format")
    parser.add_argument("--api-key", default=None,
                        help="SerpApi key (default: SERPAPI_API_KEY env var)")

    args = parser.parse_args()

    adapter = GoogleMapsAdapter(api_key=args.api_key)
    result = adapter.search_place(args.query, start=args.start, ll=args.ll)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
