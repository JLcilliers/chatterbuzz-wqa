"""
Supermetrics Google Sheets Refresh and Download Script

This script automates:
1. Opening a Google Sheet with Supermetrics queries
2. Refreshing the Supermetrics data
3. Downloading specific tabs as CSV files

Designed for use with MCP Playwright tools.
"""

import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SupermetricsConfig:
    """Configuration for Supermetrics data collection"""
    google_sheet_id: str
    tabs: List[str] = None
    output_dir: str = "."

    def __post_init__(self):
        if self.tabs is None:
            self.tabs = ["GA", "GA_prev", "GSC"]


# Tab name to output filename mapping
TAB_OUTPUT_MAP = {
    "GA": "ga_current.csv",
    "GA_prev": "ga_prev.csv",
    "GSC": "gsc_pages.csv"
}


def get_sheet_url(sheet_id: str) -> str:
    """Generate Google Sheets URL from sheet ID"""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"


# =============================================================================
# MCP PLAYWRIGHT AUTOMATION STEPS
# =============================================================================
#
# These are the step-by-step instructions for Claude to execute using
# MCP Playwright tools. Each step corresponds to a tool call.
#
# USAGE: Claude should call these MCP tools in sequence:
#
# Step 1: Navigate to the Google Sheet
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_navigate
# Parameters:
#   url: "https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit"
#
# Step 2: Wait for sheet to load
# -----------------------------------------------------------------------------
# Tool: mcp__playwright__browser_wait_for_selector
# Parameters:
#   selector: "[data-sheet-tab]" or ".docs-sheet-tab"
#   timeout: 30000
#
# Step 3: For each tab in [GA, GA_prev, GSC]:
# -----------------------------------------------------------------------------
#
#   3a. Click the tab
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "[data-sheet-tab='{tab_name}']" or text-based: "//span[text()='{tab_name}']"
#
#   3b. Trigger Supermetrics refresh (one of these methods):
#
#   Method A - Via Add-ons menu:
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "#docs-extensions-menu" or "//span[text()='Extensions']"
#
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "//span[contains(text(),'Supermetrics')]"
#
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "//span[text()='Refresh']" or "//span[text()='Refresh all']"
#
#   Method B - Via Supermetrics sidebar (if open):
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "[aria-label='Refresh']" or button with refresh icon
#
#   3c. Wait for data to load
#   Tool: mcp__playwright__browser_wait_for_selector
#   Parameters:
#     selector: Wait until no cells contain "#LOADING" or loading spinners
#     timeout: 120000  # Supermetrics can take 1-2 minutes
#
#   Alternative: Use JavaScript evaluation to check loading state:
#   Tool: mcp__playwright__browser_evaluate
#   Parameters:
#     script: |
#       // Check if any cells contain loading indicators
#       const cells = document.querySelectorAll('.cell-input');
#       return !Array.from(cells).some(c => c.textContent.includes('#LOADING'));
#
#   3d. Download the tab as CSV
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "#docs-file-menu" or "//span[text()='File']"
#
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "//span[text()='Download']"
#
#   Tool: mcp__playwright__browser_click
#   Parameters:
#     selector: "//span[text()='Comma-separated values (.csv)']"
#
#   The file will download automatically. Wait for download:
#   Tool: mcp__playwright__browser_wait_for_selector
#   Parameters:
#     selector: Wait ~5 seconds for download to complete
#
# Step 4: Rename downloaded files to expected names
# -----------------------------------------------------------------------------
# The downloaded files will have names like "{SheetName} - {TabName}.csv"
# Claude should use filesystem tools to rename them:
#   GA tab -> ga_current.csv
#   GA_prev tab -> ga_prev.csv
#   GSC tab -> gsc_pages.csv
#
# =============================================================================


AUTOMATION_INSTRUCTIONS = """
## Supermetrics Refresh and Download Automation

### Prerequisites
- Browser must be logged into Google account with access to the sheet
- Supermetrics add-on must be installed and authorized on the sheet
- Sheet must have tabs named: GA, GA_prev, GSC (or as specified)

### MCP Tool Sequence

```
1. NAVIGATE TO SHEET
   mcp__playwright__browser_navigate
   url: https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit

2. WAIT FOR SHEET TO LOAD
   mcp__playwright__browser_snapshot
   (Verify sheet tabs are visible)

3. REFRESH SUPERMETRICS DATA
   Option A - Use Extensions menu:
   - Click "Extensions" menu
   - Hover/click "Supermetrics" submenu
   - Click "Refresh all" (refreshes all tabs at once)

   Option B - Refresh each tab individually via sidebar

4. WAIT FOR DATA TO REFRESH
   - Supermetrics shows loading indicators in cells
   - Wait until all #LOADING/#GETTING_DATA cells are replaced with actual data
   - This can take 30-120 seconds depending on data volume

5. FOR EACH TAB (GA, GA_prev, GSC):
   a. Click the tab at bottom of sheet
   b. Go to File > Download > Comma-separated values (.csv)
   c. Wait for download to complete
   d. Note the downloaded filename

6. RENAME FILES
   - GA tab download -> ga_current.csv
   - GA_prev tab download -> ga_prev.csv
   - GSC tab download -> gsc_pages.csv
```

### Expected Output
```json
{
  "success": true,
  "files": {
    "ga_current": "/path/to/ga_current.csv",
    "ga_prev": "/path/to/ga_prev.csv",
    "gsc_pages": "/path/to/gsc_pages.csv"
  }
}
```

### Error Handling
- If Supermetrics menu not found: Check Extensions > Supermetrics is installed
- If refresh times out: May need to increase wait time or check Supermetrics quota
- If download fails: Check browser download settings allow automatic downloads
"""


def generate_playwright_script(config: SupermetricsConfig) -> str:
    """
    Generate a Playwright script for the automation.
    This is a reference implementation - actual execution uses MCP tools.
    """
    script = f'''
// Supermetrics Refresh and Download Script
// Generated for sheet: {config.google_sheet_id}
// Tabs to process: {config.tabs}

const SHEET_URL = "{get_sheet_url(config.google_sheet_id)}";
const TABS = {config.tabs};
const OUTPUT_DIR = "{config.output_dir}";

// Tab to filename mapping
const TAB_FILES = {{
    "GA": "ga_current.csv",
    "GA_prev": "ga_prev.csv",
    "GSC": "gsc_pages.csv"
}};

async function refreshSupermetricsAndDownload(page) {{
    // 1. Navigate to sheet
    await page.goto(SHEET_URL);
    await page.waitForSelector('[data-sheet-tab]', {{ timeout: 30000 }});

    // 2. Refresh Supermetrics (do this once for all tabs)
    // Click Extensions menu
    await page.click('#docs-extensions-menu, [aria-label="Extensions"]');
    await page.waitForTimeout(500);

    // Click Supermetrics submenu
    await page.click(':text("Supermetrics")');
    await page.waitForTimeout(500);

    // Click Refresh all
    await page.click(':text("Refresh all")');

    // 3. Wait for refresh to complete (check for loading indicators)
    await page.waitForFunction(() => {{
        const cells = document.querySelectorAll('.cell-input, .softmerge-inner');
        return !Array.from(cells).some(c =>
            c.textContent.includes('#LOADING') ||
            c.textContent.includes('#GETTING_DATA')
        );
    }}, {{ timeout: 120000 }});

    // 4. Download each tab as CSV
    const downloadedFiles = {{}};

    for (const tabName of TABS) {{
        // Click tab
        await page.click(`[data-sheet-tab="${{tabName}}"], :text("${{tabName}}")`);
        await page.waitForTimeout(1000);

        // File > Download > CSV
        await page.click('#docs-file-menu, [aria-label="File"]');
        await page.waitForTimeout(300);
        await page.click(':text("Download")');
        await page.waitForTimeout(300);

        // Start download
        const [download] = await Promise.all([
            page.waitForEvent('download'),
            page.click(':text("Comma-separated values (.csv)")')
        ]);

        // Save with correct filename
        const outputFile = TAB_FILES[tabName] || `${{tabName.toLowerCase()}}.csv`;
        const outputPath = `${{OUTPUT_DIR}}/${{outputFile}}`;
        await download.saveAs(outputPath);
        downloadedFiles[tabName] = outputPath;
    }}

    return downloadedFiles;
}}
'''
    return script


async def refresh_supermetrics_and_download_sheets(
    google_sheet_id: str,
    tabs: Optional[List[str]] = None,
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Refresh Supermetrics data and download sheets as CSV.

    This function provides the interface - actual execution happens via
    MCP Playwright tools called by Claude.

    Args:
        google_sheet_id: The Google Sheet ID containing Supermetrics queries
        tabs: List of tab names to download (default: ["GA", "GA_prev", "GSC"])
        output_dir: Directory to save downloaded CSV files

    Returns:
        Dict mapping logical names to file paths:
        {
            "ga_current": "path/to/ga_current.csv",
            "ga_prev": "path/to/ga_prev.csv",
            "gsc_pages": "path/to/gsc_pages.csv"
        }
    """
    config = SupermetricsConfig(
        google_sheet_id=google_sheet_id,
        tabs=tabs,
        output_dir=output_dir
    )

    # This is a placeholder - actual execution uses MCP tools
    # The AUTOMATION_INSTRUCTIONS above describe the MCP tool sequence

    result = {}
    for tab in config.tabs:
        output_filename = TAB_OUTPUT_MAP.get(tab, f"{tab.lower()}.csv")
        result[output_filename.replace('.csv', '')] = os.path.join(
            config.output_dir,
            output_filename
        )

    return result


# Export instructions for Claude to use
CLAUDE_INSTRUCTIONS = """
## How to Execute This Automation with MCP Playwright

When the user wants to refresh Supermetrics and download the data, follow these steps:

### Parameters Needed
- GOOGLE_SHEET_ID: The ID from the Google Sheets URL
- TABS: List of tabs to download (default: ["GA", "GA_prev", "GSC"])
- OUTPUT_DIR: Where to save CSV files (default: current directory)

### Step-by-Step MCP Tool Calls

```
1. Navigate to Sheet
   mcp__playwright__browser_navigate
   url: "https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit"

2. Take Snapshot to Verify Load
   mcp__playwright__browser_snapshot

3. Click Extensions Menu
   mcp__playwright__browser_click
   selector: "#docs-extensions-menu"
   (or use snapshot to find correct element)

4. Click Supermetrics
   mcp__playwright__browser_click
   selector: text="Supermetrics"

5. Click Refresh All
   mcp__playwright__browser_click
   selector: text="Refresh all"

6. Wait for Refresh (2 minutes max)
   mcp__playwright__browser_wait_for
   text: Wait until no #LOADING text visible
   timeout: 120

7. For Each Tab:
   a. Click tab
      mcp__playwright__browser_click
      selector: "[data-sheet-tab='GA']" (or tab name)

   b. Click File menu
      mcp__playwright__browser_click
      selector: "#docs-file-menu"

   c. Click Download
      mcp__playwright__browser_click
      selector: text="Download"

   d. Click CSV option
      mcp__playwright__browser_click
      selector: text="Comma-separated values"

   e. Wait for download
      mcp__playwright__browser_wait_for
      time: 5

8. Return file paths as JSON
```

### Notes
- Always take a snapshot after navigation to verify page state
- If selectors don't match, use snapshot to identify correct elements
- Google Sheets UI may vary - adapt selectors as needed
- Supermetrics refresh can take 30-120 seconds depending on data volume
"""
