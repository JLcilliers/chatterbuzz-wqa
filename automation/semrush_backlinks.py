"""
SEMRush Backlinks Export Script

This script automates:
1. Navigating to SEMRush Backlink Analytics → Indexed Pages
2. Exporting per-URL backlink metrics as CSV

Designed for use with MCP Playwright tools.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SEMRushBacklinkConfig:
    """Configuration for SEMRush backlink export"""
    target_domain: str
    output_dir: str = "."
    output_filename: str = "semrush_backlinks.csv"


def get_backlink_overview_url(domain: str) -> str:
    """Generate SEMRush Backlink Analytics Overview URL"""
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://www.semrush.com/analytics/backlinks/overview/?q={domain}&searchType=domain"


def get_indexed_pages_url(domain: str) -> str:
    """Generate SEMRush Backlink Analytics Indexed Pages URL"""
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://www.semrush.com/analytics/backlinks/indexedpages/?q={domain}&searchType=domain"


def get_pages_url(domain: str) -> str:
    """Alternative: Backlink Analytics Pages tab URL"""
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://www.semrush.com/analytics/backlinks/pages/?q={domain}&searchType=domain"


# =============================================================================
# MCP PLAYWRIGHT AUTOMATION STEPS
# =============================================================================
#
# These are the step-by-step instructions for Claude to execute using
# MCP Playwright tools. Each step corresponds to a tool call.
#
# USAGE: Claude should call these MCP tools in sequence:
#
# Step 1: Navigate to SEMRush Backlink Analytics
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_navigate
# Parameters:
#   url: "https://www.semrush.com/analytics/backlinks/indexedpages/?q={TARGET_DOMAIN}&searchType=domain"
#
# Alternative (if Indexed Pages doesn't exist):
#   url: "https://www.semrush.com/analytics/backlinks/overview/?q={TARGET_DOMAIN}&searchType=domain"
# Then navigate to Indexed Pages or Pages tab
#
# Step 2: Wait for page to load
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_snapshot
# (Verify the backlink data table is visible)
#
# Step 3: Navigate to Indexed Pages tab (if not there)
# -----------------------------------------------------------------------------
# SEMRush Backlink Analytics has multiple tabs:
# - Overview
# - Backlinks
# - Anchors
# - Indexed Pages (what we want - one row per URL with aggregated metrics)
# - Referring Domains
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='indexed-pages-tab']" or "//a[text()='Indexed Pages']"
#
# Step 4: Click Export button
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-button']" or "//button[contains(text(),'Export')]"
#
# Step 5: Select CSV format and All rows
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-csv']" or "//label[contains(text(),'CSV')]"
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-all-rows']" or "//label[contains(text(),'All')]"
#
# Step 6: Click Download/Export
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-download']" or "//button[text()='Export']"
#
# Step 7: Wait for download
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_wait_for
# Parameters:
#   time: 10  # Wait for file to download
#
# Step 8: Rename file to semrush_backlinks.csv
# -----------------------------------------------------------------------------
# The downloaded file may have a name like "indexed_pages_{domain}.csv"
# Rename to standardized name.
#
# =============================================================================


AUTOMATION_INSTRUCTIONS = """
## SEMRush Backlinks Export Automation

### Prerequisites
- Browser must be logged into SEMRush with appropriate subscription
- Target domain must have backlink data in SEMRush database

### What We Want
The "Indexed Pages" tab shows one row per URL with:
- Target URL (the page on the domain)
- Total Backlinks to that page
- Referring Domains pointing to that page
- Authority Score / Trust metrics (if available)

This is different from the "Backlinks" tab which shows individual backlinks.

### MCP Tool Sequence

```
1. NAVIGATE TO BACKLINK ANALYTICS INDEXED PAGES
   mcp__playwright__browser_navigate
   url: https://www.semrush.com/analytics/backlinks/indexedpages/?q={TARGET_DOMAIN}&searchType=domain

2. WAIT FOR DATA TO LOAD
   mcp__playwright__browser_snapshot
   (Verify URL list table is visible with backlink counts)

3. IF NOT ON INDEXED PAGES TAB, NAVIGATE THERE
   - Look for tab navigation in the sidebar or top nav
   - Click "Indexed Pages" tab
   mcp__playwright__browser_click
   selector: Tab or link to Indexed Pages

4. CLICK EXPORT BUTTON
   mcp__playwright__browser_click
   selector: Look for "Export" button (usually top-right of table)

5. CONFIGURE EXPORT OPTIONS
   - Select CSV format
   - Select "All" rows

6. CONFIRM EXPORT
   mcp__playwright__browser_click
   selector: "Export" or "Download" button in modal

7. WAIT FOR DOWNLOAD
   mcp__playwright__browser_wait_for
   time: 10

8. VERIFY AND RENAME FILE
   - Check download completed
   - Rename to semrush_backlinks.csv
```

### Expected Output
```json
{
  "success": true,
  "backlinks": "/path/to/semrush_backlinks.csv"
}
```

### Expected CSV Columns (Indexed Pages export)
The SEMRush Indexed Pages export should include:
- Target URL (URL on the analyzed domain)
- External Backlinks / Total Backlinks
- Referring Domains
- Dofollow Backlinks
- Nofollow Backlinks
- Authority Score (page-level)

The WQA generator specifically needs: URL, Backlinks/Referring Domains

### Alternative: Pages Tab
If "Indexed Pages" tab is not available, use the "Pages" tab which provides
similar per-URL aggregated data.

### Error Handling
- If no data shows: Domain may have no indexed pages with backlinks
- If export limited: Check SEMRush subscription level
- If tab not found: SEMRush may have renamed it - look for similar tabs
"""


def generate_playwright_script(config: SEMRushBacklinkConfig) -> str:
    """
    Generate a Playwright script for the automation.
    This is a reference implementation - actual execution uses MCP tools.
    """
    script = f'''
// SEMRush Backlinks Export Script
// Generated for domain: {config.target_domain}

const INDEXED_PAGES_URL = "{get_indexed_pages_url(config.target_domain)}";
const PAGES_URL = "{get_pages_url(config.target_domain)}";
const OUTPUT_DIR = "{config.output_dir}";
const OUTPUT_FILE = "{config.output_filename}";

async function exportSEMRushBacklinks(page) {{
    // 1. Try navigating directly to Indexed Pages
    await page.goto(INDEXED_PAGES_URL);

    // 2. Wait for page to load
    try {{
        await page.waitForSelector('[data-test="indexed-pages-table"], .backlinks-table', {{
            timeout: 30000
        }});
    }} catch (e) {{
        // If Indexed Pages doesn't load, try Pages tab
        console.log("Indexed Pages not found, trying Pages tab...");
        await page.goto(PAGES_URL);
        await page.waitForSelector('[data-test="pages-table"], .backlinks-table', {{
            timeout: 30000
        }});
    }}

    // 3. Click Export button
    await page.click('[data-test="export-button"], button:has-text("Export")');
    await page.waitForTimeout(500);

    // 4. Select CSV format
    await page.click('[data-test="export-csv"], label:has-text("CSV"), input[value="csv"]');

    // 5. Select All rows
    await page.click('[data-test="export-all"], label:has-text("All"), input[value="all"]');

    // 6. Start download
    const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-test="export-submit"], button:has-text("Export"):not([disabled])')
    ]);

    // 7. Save file
    const outputPath = `${{OUTPUT_DIR}}/${{OUTPUT_FILE}}`;
    await download.saveAs(outputPath);

    return outputPath;
}}
'''
    return script


async def export_semrush_backlinks(
    target_domain: str,
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Export SEMRush Backlink Analytics Indexed Pages data.

    This function provides the interface - actual execution happens via
    MCP Playwright tools called by Claude.

    Args:
        target_domain: The domain to analyze (e.g., "example.com")
        output_dir: Directory to save the CSV file

    Returns:
        Dict with file path:
        {
            "backlinks": "path/to/semrush_backlinks.csv"
        }
    """
    config = SEMRushBacklinkConfig(
        target_domain=target_domain,
        output_dir=output_dir
    )

    # This is a placeholder - actual execution uses MCP tools
    return {
        "backlinks": os.path.join(config.output_dir, config.output_filename)
    }


# Export instructions for Claude to use
CLAUDE_INSTRUCTIONS = """
## How to Execute SEMRush Backlinks Export with MCP Playwright

When the user wants to export SEMRush backlink data, follow these steps:

### Parameters Needed
- TARGET_DOMAIN: Domain to analyze (e.g., "example.com")
- OUTPUT_DIR: Where to save CSV (default: current directory)

### Step-by-Step MCP Tool Calls

```
1. Navigate to SEMRush Backlink Analytics - Indexed Pages
   mcp__playwright__browser_navigate
   url: "https://www.semrush.com/analytics/backlinks/indexedpages/?q={TARGET_DOMAIN}&searchType=domain"

2. Take Snapshot to Verify Page Load
   mcp__playwright__browser_snapshot
   - Verify indexed pages table is visible
   - Should show URLs with backlink counts

3. If Page Shows Error or Wrong Tab, Navigate to Correct Tab
   - Look for "Indexed Pages" or "Pages" in navigation
   mcp__playwright__browser_click
   element: "Indexed Pages tab"
   ref: (use ref from snapshot)

4. Click Export Button
   mcp__playwright__browser_click
   element: "Export button"
   ref: (use ref from snapshot)

5. In Export Modal - Select CSV
   mcp__playwright__browser_click
   element: "CSV format option"
   ref: (use ref from snapshot)

6. Select All Rows
   mcp__playwright__browser_click
   element: "All rows option"
   ref: (use ref from snapshot)

7. Click Export/Download Button
   mcp__playwright__browser_click
   element: "Export download button"
   ref: (use ref from snapshot)

8. Wait for Download
   mcp__playwright__browser_wait_for
   time: 15

9. Verify Download and Return Path
   Return: {"backlinks": "semrush_backlinks.csv"}
```

### SEMRush Backlink Analytics Tab Structure
The Backlink Analytics section has multiple tabs:
- Overview: Summary metrics
- Backlinks: Individual backlinks (one row per link) - NOT what we want
- Anchors: Anchor text analysis
- Referring Domains: Domain-level aggregation
- Indexed Pages: URL-level aggregation (WHAT WE WANT)
- Competitors: Comparison view

### Important: Indexed Pages vs Backlinks
- "Backlinks" tab = one row per backlink (source -> target)
- "Indexed Pages" tab = one row per TARGET URL with aggregate metrics

The WQA needs Indexed Pages because we want:
  URL | Total Backlinks | Referring Domains
Not individual backlink records.

### Common Selectors (may vary)
- Indexed Pages tab: `[data-test="indexed-pages-tab"]` or nav link
- Export button: `[data-test="export-button"]`
- CSV option: `[data-test="format-csv"]`
- All rows: `[data-test="rows-all"]`
- Download: `[data-test="export-submit"]`

### Fallback: Per-Backlink Export
If Indexed Pages export isn't available, the "Backlinks" tab export can be used.
The WQA generator has logic to aggregate per-backlink data by target URL:
- Counts total backlinks per URL
- Counts unique referring domains per URL
- Sums dofollow vs nofollow links

But Indexed Pages is preferred as it's already aggregated.

### Troubleshooting
- If table empty: Domain may have no backlinks in SEMRush database
- If "Indexed Pages" not visible: Try "Pages" tab as alternative
- If export limited: Check subscription level
"""


# Detailed selector hints for SEMRush Backlink Analytics page
SEMRUSH_SELECTORS = {
    # Navigation tabs
    "indexed_pages_tab": [
        "[data-test='indexed-pages-tab']",
        "a[href*='/indexedpages/']",
        "//a[contains(text(),'Indexed Pages')]",
        "//a[contains(text(),'Indexed pages')]",
        ".backlinks-nav a:has-text('Indexed')"
    ],

    "pages_tab": [
        "[data-test='pages-tab']",
        "a[href*='/pages/']",
        "//a[text()='Pages']"
    ],

    # Export button
    "export_button": [
        "[data-test='export-button']",
        "button[aria-label='Export']",
        "//button[contains(text(),'Export')]",
        ".export-button"
    ],

    # Export modal options
    "format_csv": [
        "[data-test='format-csv']",
        "input[name='format'][value='csv']",
        "//label[contains(text(),'CSV')]"
    ],

    "rows_all": [
        "[data-test='rows-all']",
        "input[name='rows'][value='all']",
        "//label[contains(text(),'All')]"
    ],

    "export_submit": [
        "[data-test='export-submit']",
        ".modal-footer button[type='submit']",
        "//button[text()='Export']"
    ],

    # Data table
    "data_table": [
        "[data-test='indexed-pages-table']",
        "[data-test='pages-table']",
        ".backlinks-table",
        "table.indexed-pages"
    ]
}
