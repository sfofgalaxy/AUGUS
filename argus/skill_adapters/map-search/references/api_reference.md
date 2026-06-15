# Map Search API Reference

## Amap (Gaode Maps) POI Search

### Endpoint
```
GET https://restapi.amap.com/v5/place/text
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| key | Yes | Amap Web Service API key |
| keywords | Yes | Search keywords |
| city | No | City name (Chinese, e.g., "上海") |
| citylimit | No | Restrict to city (true/false) |
| page_size | No | Results per page (default 20) |
| page_num | No | Page number (default 1) |
| output | No | Response format (default "json") |
| extensions | No | "all" for full details, "base" for basic |

### Response

```json
{
    "status": "1",
    "count": "42",
    "pois": [
        {
            "name": "星巴克(南京西路店)",
            "id": "B0FFHF3JCX",
            "location": "121.451234,31.234567",
            "address": "南京西路1234号",
            "pname": "上海市",
            "cityname": "上海市",
            "adname": "静安区",
            "type": "餐饮服务;咖啡厅",
            "tel": "021-12345678",
            "business_area": "南京西路",
            "photos": [{"url": "..."}]
        }
    ]
}
```

### Status Codes

| Status | Meaning |
|--------|---------|
| 1 | Success |
| 0 | Failure (check "info" field) |

### Common Info Codes

| Code | Description |
|------|-------------|
| 10000 | Success |
| 10001 | Invalid key |
| 10003 | Daily quota exceeded |
| 10044 | QPS limit exceeded |

## Google Maps (SerpApi)

### Endpoint
```
GET https://serpapi.com/search?engine=google_maps
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| engine | Yes | `"google_maps"` |
| q | Yes* | Search query (*required when type is "search", hardcoded in adapter) |
| api_key | Yes | SerpApi API key |
| ll | No | GPS coordinates: `@lat,lng,zoom` (e.g. `@40.7455096,-74.0083012,14z`). Zoom: 3z–30z. Can also use meters (e.g. `10410m`) |
| location | No | Location name for search origin. Cannot be used with `ll` or `lat`/`lon` |
| lat | No | GPS latitude. Requires `lon`. Cannot be used with `ll` or `location` |
| lon | No | GPS longitude. Requires `lat`. Cannot be used with `ll` or `location` |
| z | No | Map zoom level (3–30). Used with `location` or `lat`/`lon` |
| m | No | Map height in meters (1–15028132). Used with `location` or `lat`/`lon` |
| nearby | No | Force results closer to specified location. Recommended with "near me" queries |
| hl | No | Language code (e.g. "en", "zh-CN") |
| gl | No | Country code (e.g. "us", "cn"). Only affects Place Results |
| start | No | Pagination offset |
| place_id | No | Google Maps Place ID for a specific place |
| data_cid | No | Google CID of a place. Cannot be used with `place_id` |

### Response

```json
{
    "search_metadata": { ... },
    "local_results": [
        {
            "position": 1,
            "title": "Business Name",
            "place_id": "ChIJ...",
            "address": "Full address",
            "gps_coordinates": {
                "latitude": 31.234567,
                "longitude": 121.451234
            },
            "rating": 4.5,
            "reviews": 1234,
            "type": "Coffee shop",
            "phone": "+86-21-12345678",
            "operating_hours": { ... },
            "thumbnail": "https://..."
        }
    ]
}
```

## Coordinate Formats

| Service | Format | Example |
|---------|--------|---------|
| Amap | "longitude,latitude" | "121.451234,31.234567" |
| Google | latitude, longitude | 31.234567, 121.451234 |

Note: Amap uses GCJ-02 coordinate system; Google uses WGS-84.
