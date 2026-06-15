---
name: webpage-fetcher
description: "Fetch and extract clean text content from web pages. MUST USE when web search returned a promising URL that needs deeper reading — e.g., a LinkedIn page, company about page, university profile, or news article that could reveal identity/location details."
tools:
  - fetch_webpage
---

# Webpage Fetcher

Fetch web pages and extract clean text content, removing ads, navigation, boilerplate, and comments.

## When to Use

- Need to read full content from a URL found via web search
- Extracting article text from news sites or blogs
- Getting detailed information from a specific web page
- Verifying claims by reading source content directly
- Following up on search results that need deeper analysis

## Available Operations

1. **Fetch Webpage**: Download a URL and extract clean text content using trafilatura

## Multi-Step Workflow

### Step 1: Identify Target URL

Get URLs from:
- `web-search` results (`organic_results[].link`)
- map/search results (source page URLs)
- Direct URLs mentioned in social media posts

### Step 2: Fetch Content

```
fetch_webpage(url="https://example.com/article")
```

### Step 3: Check Response

Verify the response:
- `status`: "success" or "error"
- `content_length`: Check if content was actually extracted
- `text`: The clean extracted content

### Step 4: Analyze Content

Feed the extracted text to the analysis pipeline:
- Look for privacy-relevant information
- Cross-reference with other evidence
- Note if content was truncated (check `content_length` vs text length)

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/webpage_fetcher.py` | WebpageFetcher - trafilatura-based content extraction with truncation |

### Script CLI Usage

The script can be called directly from the command line:

```bash
# Fetch and extract text from a URL
python scripts/webpage_fetcher.py "https://example.com/privacy-policy"

# Custom timeout
python scripts/webpage_fetcher.py "https://news.site.com/article" --timeout 30

# Custom max content length
python scripts/webpage_fetcher.py "https://long-page.com" --max-length 50000
```

Output: JSON to stdout with `status`, `url`, `title`, `text`, and `content_length` fields.

## Resources (load on-demand only)

Consult references via `read_skill_file("webpage-fetcher", "references/trafilatura_guide.md")` ONLY when:
- Content extraction returns empty or garbage text and you need troubleshooting strategies
- You need to understand which content types (news, forums, JS-heavy) trafilatura handles well
- You want to understand the extraction options (include_comments, include_tables) in detail

Reference files:
- `references/trafilatura_guide.md` - Extraction options, content types, limitations, and troubleshooting

## Examples

### Example 1: Fetch Article Content

```
fetch_webpage(url="https://news.example.com/article/12345")
```
Returns:
```json
{
    "status": "success",
    "url": "https://news.example.com/article/12345",
    "title": "Article Title",
    "text": "Full article content...",
    "content_length": 5432
}
```

### Example 2: Follow Up on Search Result

After finding a relevant search result:
```
fetch_webpage(url="https://linkedin.com/in/someone")
```

## Notes

- Requires `trafilatura` library
- No JavaScript rendering (static HTML only)
- Max content length: 20,000 characters (truncated with marker)
- Content behind paywalls or requiring auth is not accessible
- Some sites may block automated requests (403 errors)
- Tables are included in extraction by default
- Best results on article/blog pages; limited on dynamic/JS-heavy sites
