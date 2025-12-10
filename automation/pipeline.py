"""
WQA Full Data Pipeline Script

This master script orchestrates all data collection:
1. Refresh Supermetrics and download GA/GSC data
2. Export SEMRush keyword rankings
3. Export SEMRush backlink data
4. Return all file paths for WQA generation

Designed for use with MCP Playwright tools.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

from .supermetrics import (
    refresh_supermetrics_and_download_sheets,
    SupermetricsConfig,
    TAB_OUTPUT_MAP
)
from .semrush_keywords import (
    export_semrush_keywords,
    SEMRushKeywordConfig
)
from .semrush_backlinks import (
    export_semrush_backlinks,
    SEMRushBacklinkConfig
)


@dataclass
class WQAPipelineConfig:
    """Configuration for the full WQA data pipeline"""
    # Required
    target_domain: str

    # Optional - Supermetrics
    google_sheet_id: Optional[str] = None
    supermetrics_tabs: List[str] = None

    # Optional - SEMRush
    semrush_database: str = "us"

    # Output
    output_dir: str = "."

    def __post_init__(self):
        if self.supermetrics_tabs is None:
            self.supermetrics_tabs = ["GA", "GA_prev", "GSC"]


@dataclass
class WQAPipelineResult:
    """Result of the full WQA data pipeline"""
    success: bool
    files: Dict[str, str]
    errors: List[str]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "files": self.files,
            "errors": self.errors
        }


# =============================================================================
# MASTER PIPELINE INSTRUCTIONS
# =============================================================================

AUTOMATION_INSTRUCTIONS = """
## WQA Full Data Pipeline Automation

This master script coordinates all data collection for a complete WQA report.

### Overview

The pipeline runs three automations in sequence:
1. **Supermetrics** - Refresh Google Sheets with GA4 and GSC data, download as CSV
2. **SEMRush Keywords** - Export Organic Research Positions data
3. **SEMRush Backlinks** - Export Backlink Analytics Indexed Pages data

### Parameters

Required:
- `TARGET_DOMAIN`: The domain to analyze (e.g., "example.com")

Optional:
- `GOOGLE_SHEET_ID`: ID of Supermetrics staging sheet (skip if not using Supermetrics)
- `SEMRUSH_DATABASE`: Country database for SEMRush (default: "us")
- `OUTPUT_DIR`: Directory for downloaded files (default: current directory)

### MCP Tool Sequence

```
========================================
PHASE 1: SUPERMETRICS (if GOOGLE_SHEET_ID provided)
========================================

1.1 Navigate to Google Sheet
    mcp__playwright__browser_navigate
    url: https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit

1.2 Take snapshot to verify load
    mcp__playwright__browser_snapshot

1.3 Refresh Supermetrics data
    - Click Extensions menu
    - Click Supermetrics submenu
    - Click "Refresh all"

1.4 Wait for refresh (up to 2 minutes)
    mcp__playwright__browser_wait_for
    text: Wait until no #LOADING cells
    time: 120

1.5 Download each tab as CSV
    For tab in [GA, GA_prev, GSC]:
      - Click tab
      - File > Download > CSV
      - Wait for download

1.6 Record file paths:
    ga_current.csv, ga_prev.csv, gsc_pages.csv

========================================
PHASE 2: SEMRUSH KEYWORDS
========================================

2.1 Navigate to Organic Research Positions
    mcp__playwright__browser_navigate
    url: https://www.semrush.com/analytics/organic/positions/?db={DATABASE}&q={TARGET_DOMAIN}&searchType=domain

2.2 Take snapshot to verify load
    mcp__playwright__browser_snapshot

2.3 Export all keyword data
    - Click Export button
    - Select CSV format
    - Select All rows
    - Click Export

2.4 Wait for download
    mcp__playwright__browser_wait_for
    time: 15

2.5 Record file path:
    semrush_keywords.csv

========================================
PHASE 3: SEMRUSH BACKLINKS
========================================

3.1 Navigate to Backlink Analytics Indexed Pages
    mcp__playwright__browser_navigate
    url: https://www.semrush.com/analytics/backlinks/indexedpages/?q={TARGET_DOMAIN}&searchType=domain

3.2 Take snapshot to verify load
    mcp__playwright__browser_snapshot

3.3 Export all indexed pages data
    - Click Export button
    - Select CSV format
    - Select All rows
    - Click Export

3.4 Wait for download
    mcp__playwright__browser_wait_for
    time: 15

3.5 Record file path:
    semrush_backlinks.csv

========================================
PHASE 4: RETURN RESULTS
========================================

Return JSON with all file paths:
{
  "success": true,
  "files": {
    "ga_current": "ga_current.csv",
    "ga_prev": "ga_prev.csv",
    "gsc_pages": "gsc_pages.csv",
    "semrush_keywords": "semrush_keywords.csv",
    "semrush_backlinks": "semrush_backlinks.csv"
  }
}
```

### Expected Output Files

| File | Source | WQA Columns Populated |
|------|--------|----------------------|
| ga_current.csv | Supermetrics GA4 | Sessions, Bounce Rate, Avg Duration, Conversions, Revenue |
| ga_prev.csv | Supermetrics GA4 | Sessions (previous period) for % Change calculation |
| gsc_pages.csv | Supermetrics GSC | Impressions, CTR, Position, Clicks |
| semrush_keywords.csv | SEMRush Organic Research | Main KW, Volume, Ranking, Best KW |
| semrush_backlinks.csv | SEMRush Backlinks | DOFOLLOW Links, Referring Domains |

### Error Handling

If any phase fails:
1. Log the error
2. Continue with remaining phases if possible
3. Return partial results with error messages

Examples:
- No Supermetrics sheet ID provided → Skip Phase 1, continue with SEMRush
- SEMRush not logged in → Return error, user must log in first
- Export times out → Retry once, then report failure

### Workflow Integration

After pipeline completes, the WQA generator should:
1. Load crawl data (user uploads Screaming Frog CSV)
2. Load pipeline output files
3. Merge all data by URL/page_path
4. Generate WQA Excel report
"""


def generate_pipeline_script(config: WQAPipelineConfig) -> str:
    """
    Generate the full pipeline script.
    This is a reference implementation - actual execution uses MCP tools.
    """
    script = f'''
// WQA Full Data Pipeline Script
// Target domain: {config.target_domain}
// Supermetrics sheet: {config.google_sheet_id or "Not configured"}
// SEMRush database: {config.semrush_database}

const CONFIG = {{
    targetDomain: "{config.target_domain}",
    googleSheetId: {f'"{config.google_sheet_id}"' if config.google_sheet_id else 'null'},
    semrushDatabase: "{config.semrush_database}",
    outputDir: "{config.output_dir}",
    supermetricsTabs: {config.supermetrics_tabs}
}};

const RESULT = {{
    success: true,
    files: {{}},
    errors: []
}};

async function runWQAPipeline(page) {{
    // =============================================
    // PHASE 1: SUPERMETRICS
    // =============================================
    if (CONFIG.googleSheetId) {{
        console.log("Phase 1: Refreshing Supermetrics data...");
        try {{
            // Navigate to sheet
            await page.goto(`https://docs.google.com/spreadsheets/d/${{CONFIG.googleSheetId}}/edit`);
            await page.waitForSelector('[data-sheet-tab]', {{ timeout: 30000 }});

            // Refresh Supermetrics
            await page.click('#docs-extensions-menu');
            await page.waitForTimeout(500);
            await page.click(':text("Supermetrics")');
            await page.waitForTimeout(500);
            await page.click(':text("Refresh all")');

            // Wait for refresh
            await page.waitForFunction(() => {{
                const cells = document.querySelectorAll('.cell-input');
                return !Array.from(cells).some(c =>
                    c.textContent.includes('#LOADING') ||
                    c.textContent.includes('#GETTING_DATA')
                );
            }}, {{ timeout: 120000 }});

            // Download each tab
            const tabFiles = {{
                "GA": "ga_current.csv",
                "GA_prev": "ga_prev.csv",
                "GSC": "gsc_pages.csv"
            }};

            for (const [tabName, fileName] of Object.entries(tabFiles)) {{
                if (CONFIG.supermetricsTabs.includes(tabName)) {{
                    await page.click(`[data-sheet-tab="${{tabName}}"]`);
                    await page.waitForTimeout(1000);

                    await page.click('#docs-file-menu');
                    await page.waitForTimeout(300);
                    await page.click(':text("Download")');
                    await page.waitForTimeout(300);

                    const [download] = await Promise.all([
                        page.waitForEvent('download'),
                        page.click(':text("Comma-separated values")')
                    ]);

                    const outputPath = `${{CONFIG.outputDir}}/${{fileName}}`;
                    await download.saveAs(outputPath);
                    RESULT.files[fileName.replace('.csv', '')] = outputPath;
                }}
            }}

            console.log("Phase 1 complete.");
        }} catch (error) {{
            console.error("Phase 1 failed:", error.message);
            RESULT.errors.push(`Supermetrics: ${{error.message}}`);
        }}
    }} else {{
        console.log("Phase 1: Skipped (no Supermetrics sheet configured)");
    }}

    // =============================================
    // PHASE 2: SEMRUSH KEYWORDS
    // =============================================
    console.log("Phase 2: Exporting SEMRush keywords...");
    try {{
        const keywordsUrl = `https://www.semrush.com/analytics/organic/positions/?db=${{CONFIG.semrushDatabase}}&q=${{CONFIG.targetDomain}}&searchType=domain`;
        await page.goto(keywordsUrl);
        await page.waitForSelector('[data-test="organic-positions-table"], .semrush-table', {{
            timeout: 30000
        }});

        await page.click('[data-test="export-button"], button:has-text("Export")');
        await page.waitForTimeout(500);
        await page.click('[data-test="export-csv"], label:has-text("CSV")');
        await page.click('[data-test="export-all"], label:has-text("All")');

        const [keywordsDownload] = await Promise.all([
            page.waitForEvent('download'),
            page.click('[data-test="export-submit"], button:has-text("Export"):not([disabled])')
        ]);

        const keywordsPath = `${{CONFIG.outputDir}}/semrush_keywords.csv`;
        await keywordsDownload.saveAs(keywordsPath);
        RESULT.files.semrush_keywords = keywordsPath;

        console.log("Phase 2 complete.");
    }} catch (error) {{
        console.error("Phase 2 failed:", error.message);
        RESULT.errors.push(`SEMRush Keywords: ${{error.message}}`);
    }}

    // =============================================
    // PHASE 3: SEMRUSH BACKLINKS
    // =============================================
    console.log("Phase 3: Exporting SEMRush backlinks...");
    try {{
        const backlinksUrl = `https://www.semrush.com/analytics/backlinks/indexedpages/?q=${{CONFIG.targetDomain}}&searchType=domain`;
        await page.goto(backlinksUrl);
        await page.waitForSelector('[data-test="indexed-pages-table"], .backlinks-table', {{
            timeout: 30000
        }});

        await page.click('[data-test="export-button"], button:has-text("Export")');
        await page.waitForTimeout(500);
        await page.click('[data-test="export-csv"], label:has-text("CSV")');
        await page.click('[data-test="export-all"], label:has-text("All")');

        const [backlinksDownload] = await Promise.all([
            page.waitForEvent('download'),
            page.click('[data-test="export-submit"], button:has-text("Export"):not([disabled])')
        ]);

        const backlinksPath = `${{CONFIG.outputDir}}/semrush_backlinks.csv`;
        await backlinksDownload.saveAs(backlinksPath);
        RESULT.files.semrush_backlinks = backlinksPath;

        console.log("Phase 3 complete.");
    }} catch (error) {{
        console.error("Phase 3 failed:", error.message);
        RESULT.errors.push(`SEMRush Backlinks: ${{error.message}}`);
    }}

    // =============================================
    // RETURN RESULTS
    // =============================================
    RESULT.success = RESULT.errors.length === 0;
    return RESULT;
}}
'''
    return script


async def run_full_wqa_data_pipeline(
    target_domain: str,
    google_sheet_id: Optional[str] = None,
    semrush_database: str = "us",
    output_dir: str = "."
) -> Dict:
    """
    Run the full WQA data collection pipeline.

    This function provides the interface - actual execution happens via
    MCP Playwright tools called by Claude.

    Args:
        target_domain: The domain to analyze (e.g., "example.com")
        google_sheet_id: Optional Supermetrics Google Sheet ID
        semrush_database: SEMRush country database (default: "us")
        output_dir: Directory for downloaded files

    Returns:
        Dict with results:
        {
            "success": True/False,
            "files": {
                "ga_current": "path/to/ga_current.csv",
                "ga_prev": "path/to/ga_prev.csv",
                "gsc_pages": "path/to/gsc_pages.csv",
                "semrush_keywords": "path/to/semrush_keywords.csv",
                "semrush_backlinks": "path/to/semrush_backlinks.csv"
            },
            "errors": []
        }
    """
    config = WQAPipelineConfig(
        target_domain=target_domain,
        google_sheet_id=google_sheet_id,
        semrush_database=semrush_database,
        output_dir=output_dir
    )

    # This is a placeholder - actual execution uses MCP tools
    files = {
        "semrush_keywords": os.path.join(output_dir, "semrush_keywords.csv"),
        "semrush_backlinks": os.path.join(output_dir, "semrush_backlinks.csv")
    }

    if google_sheet_id:
        files.update({
            "ga_current": os.path.join(output_dir, "ga_current.csv"),
            "ga_prev": os.path.join(output_dir, "ga_prev.csv"),
            "gsc_pages": os.path.join(output_dir, "gsc_pages.csv")
        })

    return {
        "success": True,
        "files": files,
        "errors": []
    }


# Export instructions for Claude to use
CLAUDE_INSTRUCTIONS = """
## How to Execute Full WQA Data Pipeline with MCP Playwright

This is the master automation that collects ALL data for a WQA report.

### Parameters
- TARGET_DOMAIN (required): Domain to analyze (e.g., "example.com")
- GOOGLE_SHEET_ID (optional): Supermetrics staging sheet ID
- SEMRUSH_DATABASE (optional): Country code for SEMRush (default: "us")
- OUTPUT_DIR (optional): Where to save files (default: current directory)

### Execution Flow

```
========================
PRE-FLIGHT CHECKS
========================
- Verify browser is logged into Google (if using Supermetrics)
- Verify browser is logged into SEMRush
- Create output directory if needed

========================
PHASE 1: SUPERMETRICS (if sheet ID provided)
========================
See: automation/supermetrics.py - CLAUDE_INSTRUCTIONS

Outcome: ga_current.csv, ga_prev.csv, gsc_pages.csv

========================
PHASE 2: SEMRUSH KEYWORDS
========================
See: automation/semrush_keywords.py - CLAUDE_INSTRUCTIONS

Outcome: semrush_keywords.csv

========================
PHASE 3: SEMRUSH BACKLINKS
========================
See: automation/semrush_backlinks.py - CLAUDE_INSTRUCTIONS

Outcome: semrush_backlinks.csv

========================
RETURN RESULTS
========================
Return JSON:
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

### Error Handling Strategy

1. **Continue on partial failure**: If one phase fails, still attempt remaining phases
2. **Collect all errors**: Track what failed for reporting
3. **Return partial results**: User can still generate WQA with available data

### Integration with WQA Generator

After pipeline completes:

1. User uploads crawl data (Screaming Frog CSV) to WQA app
2. WQA app receives pipeline output files
3. Files are parsed using existing load functions:
   - load_ga_data() for GA CSVs
   - load_gsc_data() for GSC CSV
   - load_keyword_data() for SEMRush keywords
   - load_backlink_data() for SEMRush backlinks
4. merge_datasets() combines everything
5. WQA Excel report is generated

### Timing Expectations

| Phase | Typical Duration |
|-------|-----------------|
| Supermetrics refresh | 30-120 seconds |
| Supermetrics download | 5-10 seconds per tab |
| SEMRush keywords export | 10-30 seconds |
| SEMRush backlinks export | 10-30 seconds |
| **Total** | **2-4 minutes** |

### Minimum Viable Pipeline

If Supermetrics is not configured, the pipeline can still run with just:
- SEMRush keywords
- SEMRush backlinks

User would need to upload GA/GSC data manually via CSV.
"""
