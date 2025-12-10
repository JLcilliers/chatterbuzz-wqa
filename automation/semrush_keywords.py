"""
SEMRush Keywords Export Script

This script automates:
1. Navigating to SEMRush Organic Research → Positions
2. Ensuring domain-wide scope is selected
3. Exporting all keyword positions data as CSV

Designed for use with MCP Playwright tools.
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SEMRushKeywordConfig:
    """Configuration for SEMRush keyword export"""
    target_domain: str
    database: str = "us"  # SEMRush database/country code
    output_dir: str = "."
    output_filename: str = "semrush_keywords.csv"


def get_organic_research_url(domain: str, database: str = "us") -> str:
    """Generate SEMRush Organic Research Positions URL"""
    # Clean domain (remove protocol if present)
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://www.semrush.com/analytics/organic/positions/?db={database}&q={domain}&searchType=domain"


def get_positions_tab_url(domain: str, database: str = "us") -> str:
    """Direct URL to Positions tab"""
    domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://www.semrush.com/analytics/organic/positions/?q={domain}&searchType=domain&db={database}"


# =============================================================================
# MCP PLAYWRIGHT AUTOMATION STEPS
# =============================================================================
#
# These are the step-by-step instructions for Claude to execute using
# MCP Playwright tools. Each step corresponds to a tool call.
#
# USAGE: Claude should call these MCP tools in sequence:
#
# Step 1: Navigate to SEMRush Organic Research
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_navigate
# Parameters:
#   url: "https://www.semrush.com/analytics/organic/positions/?db={DATABASE}&q={TARGET_DOMAIN}&searchType=domain"
#
# Step 2: Wait for page to load
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_snapshot
# (Verify the Organic Research page loaded with keyword data)
#
# Step 3: Verify domain scope is selected (not URL)
# -----------------------------------------------------------------------------
# The URL parameter "searchType=domain" should ensure domain-wide search.
# If not, look for a scope selector and click "Domain" option.
#
# Tool: mcp__playwright__browser_snapshot
# Check for scope selector element. If needed:
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='search-type-select']" or radio button for "Domain"
#
# Step 4: Verify Positions tab is active
# -----------------------------------------------------------------------------
# The URL should load directly to Positions tab.
# If Overview tab is shown instead, click Positions:
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='positions-tab']" or "//a[text()='Positions']"
#
# Step 5: Click Export button
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-button']" or "//button[contains(text(),'Export')]"
#
# Step 6: Select CSV format and All rows
# -----------------------------------------------------------------------------
# SEMRush export modal appears with options:
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-csv']" or "//label[contains(text(),'CSV')]"
#
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-all-rows']" or "//label[contains(text(),'All')]"
#
# Step 7: Click Download/Export
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_click
# Parameters:
#   selector: "[data-test='export-download']" or "//button[text()='Export']"
#
# Step 8: Wait for download
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_wait_for
# Parameters:
#   time: 10  # Wait for file to download
#
# Step 9: Rename file to semrush_keywords.csv
# -----------------------------------------------------------------------------
# The downloaded file may have a name like "organic_positions_{domain}.csv"
# Rename to standardized name.
#
# =============================================================================


AUTOMATION_INSTRUCTIONS = """
## SEMRush Keywords Export Automation

### Prerequisites
- Browser must be logged into SEMRush with appropriate subscription
- Target domain must have organic rankings in SEMRush database

### MCP Tool Sequence

```
1. NAVIGATE TO ORGANIC RESEARCH POSITIONS
   mcp__playwright__browser_navigate
   url: https://www.semrush.com/analytics/organic/positions/?db={DATABASE}&q={TARGET_DOMAIN}&searchType=domain

2. WAIT FOR DATA TO LOAD
   mcp__playwright__browser_snapshot
   (Verify keyword table is visible with data)

3. VERIFY SCOPE IS "DOMAIN" (not URL or subdomain)
   - Check if there's a scope selector visible
   - If set to anything other than "Domain", click to change it
   - The URL parameter searchType=domain should handle this

4. VERIFY POSITIONS TAB IS ACTIVE
   - Should be on Positions tab by default
   - If on Overview, click "Positions" tab

5. CLICK EXPORT BUTTON
   mcp__playwright__browser_click
   selector: Look for "Export" button (usually top-right of table)

6. CONFIGURE EXPORT OPTIONS
   - Select CSV format (not Excel)
   - Select "All" rows (not just visible/first 100)

7. CONFIRM EXPORT
   mcp__playwright__browser_click
   selector: "Export" or "Download" button in modal

8. WAIT FOR DOWNLOAD
   mcp__playwright__browser_wait_for
   time: 10 (or longer for large datasets)

9. VERIFY AND RENAME FILE
   - Check download completed
   - Rename to semrush_keywords.csv
```

### Expected Output
```json
{
  "success": true,
  "keywords": "/path/to/semrush_keywords.csv"
}
```

### Expected CSV Columns
The SEMRush Positions export should include:
- Keyword
- Position
- Search Volume
- URL (landing page)
- CPC
- Competition
- Number of Results
- Trends
- Keyword Difficulty
- Traffic %
- Traffic Cost %

The WQA generator specifically needs: Keyword, Position, Volume, URL

### Error Handling
- If no data shows: Domain may have no rankings in selected database
- If export limited: Check SEMRush subscription level
- If download fails: Check browser allows downloads from semrush.com
"""


def generate_playwright_script(config: SEMRushKeywordConfig) -> str:
    """
    Generate a Playwright script for the automation.
    This is a reference implementation - actual execution uses MCP tools.
    """
    script = f'''
// SEMRush Keywords Export Script
// Generated for domain: {config.target_domain}
// Database: {config.database}

const POSITIONS_URL = "{get_positions_tab_url(config.target_domain, config.database)}";
const OUTPUT_DIR = "{config.output_dir}";
const OUTPUT_FILE = "{config.output_filename}";

async function exportSEMRushKeywords(page) {{
    // 1. Navigate to Organic Research Positions
    await page.goto(POSITIONS_URL);

    // 2. Wait for data table to load
    await page.waitForSelector('[data-test="organic-positions-table"], .semrush-table', {{
        timeout: 30000
    }});

    // 3. Verify we're on Positions tab (should be by default from URL)
    const positionsTab = await page.$('[data-test="positions-tab"][aria-selected="true"], .positions-tab.active');
    if (!positionsTab) {{
        await page.click('[data-test="positions-tab"], :text("Positions")');
        await page.waitForTimeout(2000);
    }}

    // 4. Click Export button
    await page.click('[data-test="export-button"], button:has-text("Export")');
    await page.waitForTimeout(500);

    // 5. Select CSV format
    await page.click('[data-test="export-csv"], label:has-text("CSV"), input[value="csv"]');

    // 6. Select All rows
    await page.click('[data-test="export-all"], label:has-text("All"), input[value="all"]');

    // 7. Start download
    const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.click('[data-test="export-submit"], button:has-text("Export"):not([disabled])')
    ]);

    // 8. Save file
    const outputPath = `${{OUTPUT_DIR}}/${{OUTPUT_FILE}}`;
    await download.saveAs(outputPath);

    return outputPath;
}}
'''
    return script


async def export_semrush_keywords(
    target_domain: str,
    database: str = "us",
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Export SEMRush Organic Research Positions data.

    This function provides the interface - actual execution happens via
    MCP Playwright tools called by Claude.

    Args:
        target_domain: The domain to analyze (e.g., "example.com")
        database: SEMRush database/country code (default: "us")
        output_dir: Directory to save the CSV file

    Returns:
        Dict with file path:
        {
            "keywords": "path/to/semrush_keywords.csv"
        }
    """
    config = SEMRushKeywordConfig(
        target_domain=target_domain,
        database=database,
        output_dir=output_dir
    )

    # This is a placeholder - actual execution uses MCP tools
    return {
        "keywords": os.path.join(config.output_dir, config.output_filename)
    }


# Export instructions for Claude to use
CLAUDE_INSTRUCTIONS = """
## How to Execute SEMRush Keywords Export with MCP Playwright

When the user wants to export SEMRush keyword data, follow these steps:

### Parameters Needed
- TARGET_DOMAIN: Domain to analyze (e.g., "example.com")
- DATABASE: SEMRush country database (default: "us")
- OUTPUT_DIR: Where to save CSV (default: current directory)

### Step-by-Step MCP Tool Calls

```
1. Navigate to SEMRush Organic Research Positions
   mcp__playwright__browser_navigate
   url: "https://www.semrush.com/analytics/organic/positions/?db={DATABASE}&q={TARGET_DOMAIN}&searchType=domain"

2. Take Snapshot to Verify Page Load
   mcp__playwright__browser_snapshot
   - Verify keyword data table is visible
   - Check domain name matches in header

3. If Not on Positions Tab, Click It
   mcp__playwright__browser_click
   element: "Positions tab"
   ref: (use ref from snapshot)

4. Click Export Button
   mcp__playwright__browser_click
   element: "Export button"
   ref: (use ref from snapshot, usually top-right area)

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
   Return: {"keywords": "semrush_keywords.csv"}
```

### SEMRush UI Notes
- SEMRush UI uses React components with data-test attributes
- Export modal may have "All", "1000", "10000" row options depending on subscription
- Large exports may take 10-30 seconds to generate
- File downloads to browser's default download location

### Common Selectors (may vary)
- Export button: `[data-test="export-button"]` or `button:has-text("Export")`
- CSV option: `[data-test="format-csv"]` or `label:has-text("CSV")`
- All rows: `[data-test="rows-all"]` or `label:has-text("All")`
- Download: `[data-test="export-submit"]` or modal's primary button

### Troubleshooting
- If table shows "No data": Try different database (uk, de, etc.)
- If export button disabled: May need to scroll table into view first
- If modal doesn't appear: Click might have missed, retry with snapshot
"""


# Detailed selector hints for SEMRush Organic Research page
SEMRUSH_SELECTORS = {
    # Navigation
    "positions_tab": [
        "[data-test='positions-tab']",
        "a[href*='/positions/']",
        "//a[contains(text(),'Positions')]",
        ".organic-positions-tab"
    ],

    # Scope selector
    "scope_domain": [
        "[data-test='search-type-domain']",
        "input[value='domain']",
        "//label[contains(text(),'Domain')]"
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
        "[data-test='organic-positions-table']",
        ".organic-table",
        "table.positions-table"
    ]
}
