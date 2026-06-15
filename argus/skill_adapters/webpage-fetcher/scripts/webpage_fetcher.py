"""
Webpage content fetcher using trafilatura.

Fetches and extracts clean main-text content from any URL using the trafilatura
library. Strips navigation, ads, and boilerplate to return article/body text.
Includes automatic truncation to prevent overwhelming LLM context windows.

Usage as library:
    from webpage_fetcher import WebpageFetcher

    fetcher = WebpageFetcher()
    result = fetcher.fetch("https://example.com/privacy-policy")
    if result["status"] == "success":
        print(result["title"])
        print(result["text"][:500])

Usage as CLI:
    python webpage_fetcher.py "https://example.com/privacy-policy"
    python webpage_fetcher.py "https://news.site.com/article" --timeout 30
    python webpage_fetcher.py "https://long-page.com" --max-length 50000
"""

import argparse
import json
import sys
from typing import Optional


class WebpageFetcher:
    """
    Fetch and extract main text content from a webpage URL.

    Uses trafilatura for high-quality content extraction, stripping
    boilerplate (navigation, ads, footers) and retaining article text.

    Attributes:
        MAX_CONTENT_LENGTH: Maximum characters returned (default 20000).
    """

    MAX_CONTENT_LENGTH = 20000  # chars, avoid feeding too much to LLM

    def __init__(self, max_content_length: int | None = None) -> None:
        if max_content_length is not None:
            self.MAX_CONTENT_LENGTH = max_content_length

    def fetch(self, url: str, timeout: int = 15) -> dict:
        """
        Fetch a URL and return cleaned text content using trafilatura.

        Args:
            url: The webpage URL to fetch.
            timeout: Request timeout in seconds.

        Returns:
            dict with keys:
                status: "success" | "error"
                url: the fetched URL
                title: page title (if found)
                text: extracted text content (truncated to MAX_CONTENT_LENGTH)
                content_length: original text length before truncation
                message: error message (only when status == "error")
        """
        try:
            import trafilatura

            # Download page
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {
                    "status": "error",
                    "url": url,
                    "title": "",
                    "text": "",
                    "content_length": 0,
                    "message": "Failed to fetch URL",
                }

            # Extract main content
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            ) or ""

            # Extract title
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else ""

            original_length = len(text)
            if len(text) > self.MAX_CONTENT_LENGTH:
                text = text[: self.MAX_CONTENT_LENGTH] + "\n...[truncated]"

            return {
                "status": "success",
                "url": url,
                "title": title,
                "text": text,
                "content_length": original_length,
            }

        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "title": "",
                "text": "",
                "content_length": 0,
                "message": str(e),
            }


def main():
    """CLI entry point for webpage content extraction."""
    parser = argparse.ArgumentParser(
        description="Extract main text content from a webpage using trafilatura",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python webpage_fetcher.py 'https://example.com/privacy-policy'",
    )
    parser.add_argument("url", help="URL of the webpage to fetch")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Request timeout in seconds (default: 15)")
    parser.add_argument("--max-length", type=int, default=20000,
                        help="Max content length in chars (default: 20000)")

    args = parser.parse_args()

    fetcher = WebpageFetcher(max_content_length=args.max_length)
    result = fetcher.fetch(args.url, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
