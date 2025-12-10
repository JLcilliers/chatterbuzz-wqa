# WQA Data Collection - MCP Playwright Execution Guide

This guide provides step-by-step instructions for Claude to execute WQA data collection automations using MCP Playwright tools.

## Prerequisites

Before running any automation:
1. Browser must be logged into Google (for Supermetrics)
2. Browser must be logged into SEMRush
3. Download directory should be accessible

## Quick Reference

| Automation | Input Parameters | Output Files |
|------------|-----------------|--------------|
| Supermetrics | `GOOGLE_SHEET_ID`, `TABS` | `ga_current.csv`, `ga_prev.csv`, `gsc_pages.csv` |
| SEMRush Keywords | `TARGET_DOMAIN`, `DATABASE` | `semrush_keywords.csv` |
| SEMRush Backlinks | `TARGET_DOMAIN` | `semrush_backlinks.csv` |
| Full Pipeline | All above | All above |

---

## 1. Supermetrics Refresh & Download

### Parameters
- `GOOGLE_SHEET_ID`: e.g., `"1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"`
- `TABS`: e.g., `["GA", "GA_prev", "GSC"]`

### MCP Tool Sequence

```
STEP 1: Navigate to Sheet
─────────────────────────
Tool: mcp__playwright__browser_navigate
Params:
  url: "https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit"


STEP 2: Verify Sheet Loaded
───────────────────────────
Tool: mcp__playwright__browser_snapshot
Expected: Sheet tabs visible at bottom


STEP 3: Open Extensions Menu
────────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Extensions menu button"
  ref: [get from snapshot - usually has id "docs-extensions-menu"]


STEP 4: Click Supermetrics
──────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Supermetrics menu item"
  ref: [from snapshot - submenu item containing "Supermetrics"]


STEP 5: Click Refresh All
─────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Refresh all option"
  ref: [from snapshot - menu item "Refresh all" or "Refresh"]


STEP 6: Wait for Data Refresh
─────────────────────────────
Tool: mcp__playwright__browser_wait_for
Params:
  time: 90  # Supermetrics can take 30-90 seconds

Then take snapshot to verify no #LOADING cells remain.


STEP 7: For Each Tab (GA, GA_prev, GSC):
────────────────────────────────────────

  7a. Click tab at bottom
  Tool: mcp__playwright__browser_click
  Params:
    element: "{TAB_NAME} tab"
    ref: [sheet tab at bottom of page]

  7b. Click File menu
  Tool: mcp__playwright__browser_click
  Params:
    element: "File menu"
    ref: [usually id "docs-file-menu"]

  7c. Click Download
  Tool: mcp__playwright__browser_click
  Params:
    element: "Download submenu"
    ref: [menu item "Download"]

  7d. Click CSV option
  Tool: mcp__playwright__browser_click
  Params:
    element: "Comma-separated values (.csv)"
    ref: [submenu item]

  7e. Wait for download
  Tool: mcp__playwright__browser_wait_for
  Params:
    time: 5


STEP 8: Return Results
──────────────────────
{
  "ga_current": "ga_current.csv",
  "ga_prev": "ga_prev.csv",
  "gsc_pages": "gsc_pages.csv"
}
```

---

## 2. SEMRush Keywords Export

### Parameters
- `TARGET_DOMAIN`: e.g., `"example.com"`
- `DATABASE`: e.g., `"us"` (country code)

### MCP Tool Sequence

```
STEP 1: Navigate to Organic Research Positions
──────────────────────────────────────────────
Tool: mcp__playwright__browser_navigate
Params:
  url: "https://www.semrush.com/analytics/organic/positions/?db={DATABASE}&q={TARGET_DOMAIN}&searchType=domain"


STEP 2: Verify Page Loaded
──────────────────────────
Tool: mcp__playwright__browser_snapshot
Expected: Keyword data table visible with columns like:
  - Keyword
  - Position
  - Volume
  - URL


STEP 3: Click Export Button
───────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Export button"
  ref: [from snapshot - button in toolbar area, typically has "Export" text]


STEP 4: Select CSV Format
─────────────────────────
Tool: mcp__playwright__browser_snapshot
(Get modal contents)

Tool: mcp__playwright__browser_click
Params:
  element: "CSV format radio/checkbox"
  ref: [from snapshot - option labeled "CSV"]


STEP 5: Select All Rows
───────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "All rows option"
  ref: [from snapshot - option labeled "All" for row count]


STEP 6: Click Export/Download
─────────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Export submit button"
  ref: [from snapshot - primary button in modal, usually labeled "Export"]


STEP 7: Wait for Download
─────────────────────────
Tool: mcp__playwright__browser_wait_for
Params:
  time: 15  # Large keyword sets may take longer


STEP 8: Return Results
──────────────────────
{
  "keywords": "semrush_keywords.csv"
}
```

---

## 3. SEMRush Backlinks Export

### Parameters
- `TARGET_DOMAIN`: e.g., `"example.com"`

### MCP Tool Sequence

```
STEP 1: Navigate to Backlink Analytics - Indexed Pages
──────────────────────────────────────────────────────
Tool: mcp__playwright__browser_navigate
Params:
  url: "https://www.semrush.com/analytics/backlinks/indexedpages/?q={TARGET_DOMAIN}&searchType=domain"


STEP 2: Verify Page Loaded
──────────────────────────
Tool: mcp__playwright__browser_snapshot
Expected: Table showing URLs with:
  - Target URL
  - Backlinks count
  - Referring Domains count


STEP 3: (If needed) Navigate to Indexed Pages Tab
─────────────────────────────────────────────────
If not on Indexed Pages view:
Tool: mcp__playwright__browser_click
Params:
  element: "Indexed Pages tab"
  ref: [from snapshot - navigation tab/link]


STEP 4: Click Export Button
───────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Export button"
  ref: [from snapshot - button in toolbar]


STEP 5: Select CSV Format
─────────────────────────
Tool: mcp__playwright__browser_snapshot
(Get modal)

Tool: mcp__playwright__browser_click
Params:
  element: "CSV format option"
  ref: [from snapshot]


STEP 6: Select All Rows
───────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "All rows option"
  ref: [from snapshot]


STEP 7: Click Export/Download
─────────────────────────────
Tool: mcp__playwright__browser_click
Params:
  element: "Export submit button"
  ref: [from snapshot]


STEP 8: Wait for Download
─────────────────────────
Tool: mcp__playwright__browser_wait_for
Params:
  time: 15


STEP 9: Return Results
──────────────────────
{
  "backlinks": "semrush_backlinks.csv"
}
```

---

## 4. Full WQA Data Pipeline

### Parameters
- `TARGET_DOMAIN`: Required
- `GOOGLE_SHEET_ID`: Optional (skip Supermetrics if not provided)
- `DATABASE`: Optional (default: "us")

### MCP Tool Sequence

```
PHASE 1: SUPERMETRICS (if GOOGLE_SHEET_ID provided)
═══════════════════════════════════════════════════
Run Supermetrics automation (Section 1 above)
Result: ga_current.csv, ga_prev.csv, gsc_pages.csv


PHASE 2: SEMRUSH KEYWORDS
═════════════════════════
Run SEMRush Keywords automation (Section 2 above)
Result: semrush_keywords.csv


PHASE 3: SEMRUSH BACKLINKS
══════════════════════════
Run SEMRush Backlinks automation (Section 3 above)
Result: semrush_backlinks.csv


PHASE 4: RETURN COMBINED RESULTS
════════════════════════════════
{
  "success": true,
  "files": {
    "ga_current": "ga_current.csv",
    "ga_prev": "ga_prev.csv",
    "gsc_pages": "gsc_pages.csv",
    "semrush_keywords": "semrush_keywords.csv",
    "semrush_backlinks": "semrush_backlinks.csv"
  },
  "errors": []
}
```

---

## Troubleshooting

### Supermetrics Issues

| Problem | Solution |
|---------|----------|
| Extensions menu not found | Try alternative selectors: `[aria-label="Extensions"]`, `//span[text()='Extensions']` |
| Supermetrics not in menu | Add-on may not be installed; user needs to install it |
| Refresh times out | Increase wait time to 120-180 seconds; large datasets take longer |
| Download fails | Check browser download settings; may need to handle download dialog |

### SEMRush Issues

| Problem | Solution |
|---------|----------|
| No data shown | Try different database (uk, de, fr, etc.); domain may have no rankings |
| Export button disabled | Scroll page or table into view; button may be conditionally enabled |
| "All rows" option missing | Subscription may limit export size; use available max option |
| Login redirect | Browser session expired; user needs to log in manually |

### General Issues

| Problem | Solution |
|---------|----------|
| Selector not found | Take fresh snapshot; UI may have changed |
| Click doesn't work | Try alternative selectors; use snapshot ref values |
| Download location unknown | Check browser's download directory; may need configuration |

---

## Expected CSV Schemas

### ga_current.csv / ga_prev.csv
```csv
pagePath,sessions,bounceRate,averageSessionDuration,keyEvents,purchaseRevenue
/,5000,0.45,120.5,150,25000.00
/products/,2500,0.52,95.3,75,12000.00
```

### gsc_pages.csv
```csv
page,impressions,clicks,ctr,position
https://example.com/,100000,5000,0.05,3.2
https://example.com/products/,50000,2000,0.04,5.1
```

### semrush_keywords.csv
```csv
Keyword,Position,Search Volume,URL,CPC,Competition
best product,3,5400,https://example.com/products/,2.50,0.85
buy product online,7,2900,https://example.com/,1.80,0.72
```

### semrush_backlinks.csv
```csv
Target URL,External Links,Referring Domains,Dofollow,Nofollow
https://example.com/,15420,1250,12000,3420
https://example.com/products/,3200,450,2800,400
```

---

## Integration with WQA Generator

After collecting all CSV files, feed them to the WQA generator:

1. **Crawl Data**: User uploads Screaming Frog/Sitebulb CSV
2. **GA Data**: Use `ga_current.csv` and `ga_prev.csv`
3. **GSC Data**: Use `gsc_pages.csv`
4. **Keywords**: Use `semrush_keywords.csv`
5. **Backlinks**: Use `semrush_backlinks.csv`

The WQA generator's existing functions handle all parsing and merging:
- `load_ga_data()` - Parses GA CSVs
- `load_gsc_data()` - Parses GSC CSV
- `load_keyword_data()` - Parses keyword CSV, aggregates by URL
- `load_backlink_data()` - Parses backlink CSV, handles both formats
- `merge_datasets()` - Joins all data by URL

Final output: Complete WQA Excel report with all columns populated.
