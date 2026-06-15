# Trafilatura Reference Guide

## Overview

Trafilatura is a Python library for web content extraction. It downloads web pages and extracts the main text content, removing ads, navigation, boilerplate, and comments.

## How It Works

1. **Fetch**: Downloads the HTML page via `trafilatura.fetch_url(url)`
2. **Extract**: Parses HTML and identifies main content via `trafilatura.extract()`
3. **Metadata**: Extracts page title via `trafilatura.extract_metadata()`

## Extraction Options

| Option | Default | Description |
|--------|---------|-------------|
| `include_comments` | False | Include user comments in output |
| `include_tables` | True | Include table content |
| `no_fallback` | False | If True, skip backup extraction methods |

## Content Truncation

The adapter enforces a `MAX_CONTENT_LENGTH` of 20,000 characters. Content exceeding this limit is truncated with a `...[truncated]` marker. The original length is returned in the `content_length` field.

## Response Format

```json
{
    "status": "success",
    "url": "https://example.com/article",
    "title": "Page Title",
    "text": "Extracted main content...",
    "content_length": 15432
}
```

On error:
```json
{
    "status": "error",
    "url": "https://example.com/article",
    "title": "",
    "text": "",
    "content_length": 0,
    "message": "Error description"
}
```

## Supported Content Types

- News articles and blog posts (best results)
- Documentation pages
- Forum posts and discussions
- Product pages (partial)
- Dynamic/JavaScript-heavy pages (limited - only static HTML is processed)

## Limitations

- No JavaScript rendering (static HTML only)
- May fail on pages requiring authentication
- Content behind paywalls is not accessible
- Very large pages may be slow to process
- Some sites block automated requests (403 errors)

## Tips for Best Results

1. Use direct article URLs rather than index/listing pages
2. If extraction fails, the URL may require JavaScript rendering
3. Check `content_length` - very short results may indicate extraction issues
4. Tables are included by default; disable if not needed for cleaner output
