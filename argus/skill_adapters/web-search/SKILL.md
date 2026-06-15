---
name: web-search
description: "Search the web using Google or Bing via SerpApi. MUST USE when: OCR/caption/visual analysis contains person names, business names, product models, event names, or any entity that needs verification. This is your PRIMARY verification tool — if you have an unverified entity, search it."
tools:
  - google_search
  - bing_search
---

# Web Search

Search the web using Google or Bing engines via SerpApi for information gathering and verification.

## When to Use — Privacy Investigation Triggers

**ALWAYS search when you encounter these in OCR/caption/VL evidence:**
- Person names, @mentions, or usernames → search to find identity info
- Business names or brand names → search to verify location and details
- Product model numbers (e.g., "iPhone 16 Pro", "Tesla Model Y") → search for price/details
- Event names (concerts, conferences, exhibitions) → search for date/venue/location
- University/school/lab names → search for education details
- Company names or job titles → search for employer info
- Addresses or landmarks → search to verify and geolocate
- Any entity surfaced by OCR, caption, or visual analysis

**ALWAYS search when you need to:**
- Cross-reference an OCR finding with public information
- Verify a location hypothesis from map search
- Follow up on named entities surfaced from visual analysis
- Check if a person/business is publicly known

## When NOT to Use

- The information is an obvious common fact (e.g., "Beijing is in China")
- You already verified the same entity in a previous tool call
- The claim is directly and explicitly stated in the caption with no ambiguity

## Available Operations

1. **Google Web Search**: Full-featured web search with pagination (English results)
2. **Bing Web Search**: Alternative search engine with country-specific filtering

## Multi-Step Workflow

### Step 1: Determine Search Strategy

Choose the appropriate search engine based on the target:
- **Google** (`google_search`): Best for general queries, English content, international results
- **Bing** (`bing_search`): Alternative perspective, useful when Google results are insufficient

### Step 2: Construct Query

Build an effective search query:
- Start specific: use exact names, addresses, or identifiers
- Add geographic context (e.g., city names, country)
- Use quotes for exact phrase matching

### Step 3: Execute Search

Call the appropriate tool:
```
google_search(query="Starbucks Reserve Roastery Shanghai", num=5)
bing_search(query="Tokyo Tower observation deck", num=5, cc="jp")
```

### Step 4: Analyze Results

Parse the JSON response:
- Check `status` field for success/error
- Extract relevant info from `organic_results`
- If results are insufficient, broaden query or try alternate engine

### Step 5: Follow Up (Optional)

For promising results, use `fetch_webpage` to get full page content from result URLs.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/google_search.py` | GoogleSearchAdapter - SerpApi Google Web Search (English, hl=en) |
| `scripts/bing_search.py` | BingSearchAdapter - SerpApi Bing Web Search with country filtering |

### Script CLI Usage

Scripts can be called directly from the command line for quick testing or standalone use:

```bash
# Google search
python scripts/google_search.py "privacy policy analysis" --num 10
python scripts/google_search.py "data breach" --start 10 --api-key YOUR_KEY

# Bing search
python scripts/bing_search.py "cookie consent banner" --num 10 --cc gb
python scripts/bing_search.py "GDPR compliance" --first 11

# Using environment variable for API key
SERPAPI_API_KEY=xxx python scripts/google_search.py "search query"
```

Output: JSON to stdout with `status`, `raw`, and `organic_results` fields.

## Resources (load on-demand only)

Consult references via `read_skill_file("web-search", "references/api_reference.md")` ONLY when:
- You get an unexpected error or empty results and need to check parameter constraints
- You need advanced parameters not shown in the examples above (e.g., `safe`, `mkt`)
- You need to understand the full response JSON structure for parsing specific fields

Reference files:
- `references/api_reference.md` - SerpApi parameter reference, response format details, and rate limit information

## Tool Chain Patterns (Privacy Investigation)

### Chain 1: OCR → Web Search → Verify
```
1. OCR extracts: "星巴克烘焙工坊"
2. google_search(query="星巴克烘焙工坊 地址", num=5)
3. Result confirms: Shanghai, Nanjing Road → location evidence
```

### Chain 2: Visual Entity → Web Search → Fetch
```
1. deep_visual_analysis or OCR surfaces: "universityX"
2. google_search(query="universityX department faculty", num=5)
3. fetch_webpage on relevant result → extract details
```

### Chain 3: Map Search → Web Search (cross-verify)
```
1. amap_poi_search finds: "某某餐厅, 杭州市西湖区"
2. google_search(query="某某餐厅 杭州 评价", num=5)
3. Cross-check address consistency
```

### Failure Recovery
- Google returns irrelevant results → try Bing with `cc` parameter
- Too few results → broaden query (remove city, use fewer keywords)
- Results in wrong language → add language hint to query

## Examples

### Example 1: Location Verification

Verify a business location found in a social media post:
```
google_search(query="瑞幸咖啡 南京西路 上海", num=5)
```
Expected: JSON with organic_results containing business address and details.

### Example 2: Person/Entity Lookup

OCR found "@username" or a person name in caption:
```
google_search(query="张三 清华大学 计算机", num=5)
```

### Example 3: Event Verification

Caption mentions a concert or conference:
```
google_search(query="ICRA 2024 conference venue date location", num=5)
```

### Example 4: Product Identification

OCR detected a model number:
```
google_search(query="Canon EOS R5 price specifications", num=5)
```

## Notes

- Requires `SERPAPI_API_KEY` in environment
- Rate limits apply per SerpApi account plan
- Google: results truncated client-side by `num`; Bing: results truncated client-side by `num`
- Timeout: 30 seconds per request
- Results returned as-is from SerpApi (JSON format)
- Bing: use `cc` for country-specific results (e.g., us, jp, gb, kr)
