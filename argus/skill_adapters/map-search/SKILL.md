---
name: map-search
description: "Search for places/POIs using Amap (China) or Google Maps (international). MUST USE when: OCR/caption contains road names (路/街/道/avenue/road), store/restaurant names, metro stations, landmarks, navigation UI text (导航/出口/限速), or any place name that needs geolocation. This pins down the user's location."
tools:
  - amap_poi_search
  - google_maps_search
---

# Map Search

Search for places, businesses, and points of interest using region-appropriate map providers.

## When to Use — Privacy Investigation Triggers

**ALWAYS use map search when OCR/caption/VL evidence contains:**
- Road or street names: 路, 街, 道, 大道, avenue, road, street, boulevard
- Store/restaurant/business names with location context
- Metro/subway station names (地铁站, metro, station)
- Navigation UI content: 导航, 出口, 限速, HUD, 公里, 左转, 右转
- Highway/expressway names: 高速, 出口, expressway
- Landmarks: 广场, plaza, square, tower, park, 公园
- Address fragments: district names, building numbers

**Provider selection:**
- Chinese text or Chinese cities → `amap_poi_search` (much better for China POI data)
- International content → `google_maps_search`
- Uncertain → try Amap first, then Google Maps

## When NOT to Use

- Location is already verified by a previous map search
- The place name is too generic without any geographic context (e.g., just "coffee shop")
- You only have a country-level location (use web search instead)

## Available Operations

1. **Amap POI Search**: Search for places in China using Gaode Maps API
2. **Google Maps Search**: Search for places internationally via SerpApi

## Multi-Step Workflow

### Step 1: Determine Region

Choose the appropriate provider:
- **Chinese content** (Chinese text, Chinese cities) → Use `amap_poi_search`
- **International content** → Use `google_maps_search`
- **Uncertain** → Try Amap first for Chinese names, then Google Maps

### Step 2: Construct Search Query

- Use specific business names or addresses when available
- Add city context to narrow results
- For Amap: use Chinese keywords for best results

### Step 3: Execute Search

```
amap_poi_search(keyword="星巴克烘焙工坊", city="上海")
google_maps_search(query="Starbucks Reserve Roastery", ll="@31.23,121.47,15z")
```

### Step 4: Analyze Results

- Check location details (address, coordinates, business info)
- Verify against other evidence (photos, text mentions)
- Note discrepancies for the verifier

### Step 5: Cross-Reference

If results are inconclusive:
- Try the other provider
- Broaden the search (remove city restriction)
- Combine with `web-search` for additional context

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/amap_adapter.py` | AmapAdapter - Gaode Maps POI search for China (v5 API) |
| `scripts/gmaps_adapter.py` | GoogleMapsAdapter - Google Maps Local Results via SerpApi |

### Script CLI Usage

Scripts can be called directly from the command line:

```bash
# Amap POI search (China)
python scripts/amap_adapter.py "星巴克" --city 北京 --page-size 10
python scripts/amap_adapter.py "数据中心" --city 深圳 --no-citylimit
AMAP_API_KEY=xxx python scripts/amap_adapter.py "南京大学"

# Google Maps search (international)
python scripts/gmaps_adapter.py "coffee shop" --ll "@40.7128,-74.0060,15z"
python scripts/gmaps_adapter.py "Tokyo Tower" --start 20
SERPAPI_API_KEY=xxx python scripts/gmaps_adapter.py "privacy consultant"
```

Output: JSON to stdout with POI details (name, address, coordinates, business info).

## Resources (load on-demand only)

Consult references via `read_skill_file("map-search", "references/api_reference.md")` ONLY when:
- You get Amap error codes (e.g., infocode != 10000) and need to diagnose the issue
- You need coordinate format details (GCJ-02 vs WGS-84 differences)
- You need advanced Google Maps parameters (e.g., `location`, `lat`/`lon`, `nearby`, `place_id`, `hl`, `gl`)

Reference files:
- `references/api_reference.md` - API parameters, response formats, and region-specific tips

## Tool Chain Patterns (Privacy Investigation)

### Chain 1: OCR (navigation UI) → Map Search → Location Inference
```
1. OCR extracts: "沪昆高速 出口 3.2公里"
2. amap_poi_search(keyword="沪昆高速", city=None)
3. Result → narrows user's driving location to specific highway segment
4. Combine with other location evidence for home area inference
```

### Chain 2: OCR (store name) → Map Search → Web Search
```
1. OCR extracts: "鼎泰丰 环球金融中心店"
2. amap_poi_search(keyword="鼎泰丰 环球金融中心", city="上海")
3. Confirms: Pudong, Shanghai → location evidence
4. Optional: google_search for more context
```

### Failure Recovery
- No results → remove city restriction, search broader
- Multiple matches → add more context keywords from OCR
- Wrong region → try the other provider (Amap ↔ Google Maps)

## Examples

### Example 1: Chinese Location

```
amap_poi_search(keyword="南京大学鼓楼校区", city="南京")
```

### Example 2: Navigation Clues

OCR found navigation UI text:
```
amap_poi_search(keyword="京港澳高速 岳阳出口")
```

### Example 3: International Location

```
google_maps_search(query="Tokyo Tower", ll="@35.66,139.75,14z")
```

### Example 4: Broad Search (No City)

```
amap_poi_search(keyword="西湖国宾馆")
```

## Notes

- Amap requires `AMAP_API_KEY`; Google Maps requires `SERPAPI_API_KEY`
- Amap is significantly better for Chinese POI data
- Google Maps returns richer international data
- Amap timeout: 10 seconds; Google Maps timeout: 30 seconds
- Amap returns up to 20 results per page with pagination support
