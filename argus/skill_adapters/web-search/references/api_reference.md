# SerpApi Web Search Reference

## Base URL

```
https://serpapi.com/search
```

## Google Search Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| engine | Yes | Set to `"google"` |
| q | Yes | Search query string |
| api_key | Yes | Your SerpApi API key |
| start | No | Result offset for pagination (default 0) |
| hl | No | Language code (default "en"). Fixed to "en" in adapter |
| location | No | Search origin location (city-level recommended) |
| google_domain | No | Google domain (e.g., "google.com") |
| gl | No | Country code for localized results (e.g., "us", "cn") |
| safe | No | Safe search: "active" or "off" |
| tbs | No | Advanced search filters (date range, etc.) |

## Bing Search Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| engine | Yes | Set to `"bing"` |
| q | Yes | Search query string |
| api_key | Yes | Your SerpApi API key |
| first | No | Result start position (default 1) |
| cc | No | Country code (e.g., "us", "cn"). Mutually exclusive with `mkt` |
| mkt | No | Market code (e.g., "en-US"). Mutually exclusive with `cc` |
| safeSearch | No | Adult content filter: "Off", "Moderate", or "Strict" |

## Response Structure

### Success Response

```json
{
    "search_metadata": {
        "status": "Success",
        "id": "...",
        "created_at": "..."
    },
    "organic_results": [
        {
            "position": 1,
            "title": "Page Title",
            "link": "https://example.com/page",
            "snippet": "Brief description of the page content...",
            "displayed_link": "example.com",
            "source": "Example.com"
        }
    ],
    "knowledge_graph": { ... },
    "related_searches": [ ... ]
}
```

### Error Response

```json
{
    "error": "Error message describing what went wrong"
}
```

## Rate Limits

- Free tier: 250 searches/month
- Paid plans: varies by subscription
- 429 status code indicates rate limit exceeded

## Tips

- Results are truncated client-side via `num` parameter in the adapter
- Google adapter uses `hl=en` by default for English results
- Bing: use `cc` for country-specific results
- Check `knowledge_graph` for structured information about entities
- Use `related_searches` to discover alternative query terms
