#!/usr/bin/env python3
"""
Verification script for Analytical Insight sheets in WQA Generator output.

This script verifies that the three analytical insight sheets are correctly created:
- Content to Optimize
- Thin Content Opportunities
- New Content Opportunities

Usage:
    python test_analytical_sheets.py <path_to_wqa_output.xlsx>

Example:
    python test_analytical_sheets.py wqa_report.xlsx
"""

import sys
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip install openpyxl")
    sys.exit(1)


def verify_analytical_sheets(excel_path: str) -> bool:
    """
    Verify that the analytical insight sheets exist and have correct structure.

    Args:
        excel_path: Path to the WQA output Excel file

    Returns:
        True if all verifications pass, False otherwise
    """
    errors = []
    warnings = []

    # Check file exists
    if not Path(excel_path).exists():
        print(f"ERROR: File not found: {excel_path}")
        return False

    # Load workbook
    try:
        wb = load_workbook(excel_path, read_only=False)
    except Exception as e:
        print(f"ERROR: Could not load workbook: {e}")
        return False

    print(f"Loaded workbook: {excel_path}")
    print(f"Available sheets: {wb.sheetnames}")
    print()

    # ==========================================================================
    # Verify Sheet 1: Content to Optimize
    # ==========================================================================
    sheet_name = 'Content to Optimize'
    print(f"Checking '{sheet_name}'...")

    if sheet_name not in wb.sheetnames:
        errors.append(f"Sheet '{sheet_name}' not found")
    else:
        ws = wb[sheet_name]
        print(f"  [OK] Sheet exists")

        # Check for expected headers
        expected_headers = [
            'URL', 'Page Type', 'Avg Position', 'Impressions', 'Clicks', 'CTR (%)',
            'Word Count', 'Has Meta Description', 'Sessions', 'Optimization Signals', 'Priority Score'
        ]

        # Check if sheet has data or the "no opportunities" message
        first_cell = ws.cell(row=1, column=1).value
        if first_cell == 'No content optimization opportunities found':
            print(f"  [OK] Sheet has 'no opportunities' message (data-dependent)")
        else:
            # Check headers
            for col_idx, expected in enumerate(expected_headers, start=1):
                actual = ws.cell(row=1, column=col_idx).value
                if actual != expected:
                    errors.append(f"  {sheet_name}: Column {col_idx} header - expected '{expected}', got '{actual}'")
                else:
                    print(f"  [OK] Column {col_idx}: '{expected}'")

            # Check freeze panes
            if ws.freeze_panes != 'A2':
                warnings.append(f"  {sheet_name}: Freeze panes expected 'A2', got '{ws.freeze_panes}'")
            else:
                print(f"  [OK] Freeze panes at A2")

            # Count data rows (should be capped at max 20)
            row_count = ws.max_row - 1  # Exclude header
            print(f"  [OK] Data rows: {row_count}")

            # Verify Top 20 cap
            if row_count > 20:
                errors.append(f"  {sheet_name}: Expected max 20 rows, got {row_count}")
            else:
                print(f"  [OK] Row count within Top 20 cap")

    print()

    # ==========================================================================
    # Verify Sheet 2: Thin Content Opportunities
    # ==========================================================================
    sheet_name = 'Thin Content Opportunities'
    print(f"Checking '{sheet_name}'...")

    if sheet_name not in wb.sheetnames:
        errors.append(f"Sheet '{sheet_name}' not found")
    else:
        ws = wb[sheet_name]
        print(f"  [OK] Sheet exists")

        # Check for expected headers
        expected_headers = [
            'URL', 'Page Type', 'Word Count', 'Content Gap', 'Sessions',
            'Impressions', 'Avg Position', 'Referring Domains', 'Opportunity Type', 'Value Score'
        ]

        # Check if sheet has data or the "no opportunities" message
        first_cell = ws.cell(row=1, column=1).value
        if first_cell == 'No thin content opportunities found':
            print(f"  [OK] Sheet has 'no opportunities' message (data-dependent)")
        else:
            # Check headers
            for col_idx, expected in enumerate(expected_headers, start=1):
                actual = ws.cell(row=1, column=col_idx).value
                if actual != expected:
                    errors.append(f"  {sheet_name}: Column {col_idx} header - expected '{expected}', got '{actual}'")
                else:
                    print(f"  [OK] Column {col_idx}: '{expected}'")

            # Check freeze panes
            if ws.freeze_panes != 'A2':
                warnings.append(f"  {sheet_name}: Freeze panes expected 'A2', got '{ws.freeze_panes}'")
            else:
                print(f"  [OK] Freeze panes at A2")

            # Count data rows
            row_count = ws.max_row - 1  # Exclude header
            print(f"  [OK] Data rows: {row_count}")

    print()

    # ==========================================================================
    # Verify Sheet 3: New Content Opportunities
    # ==========================================================================
    sheet_name = 'New Content Opportunities'
    print(f"Checking '{sheet_name}'...")

    if sheet_name not in wb.sheetnames:
        errors.append(f"Sheet '{sheet_name}' not found")
    else:
        ws = wb[sheet_name]
        print(f"  [OK] Sheet exists")

        # Check for expected headers (topic-based, not URL-based)
        # Note: Cannibalization Risk column added between Avg Position and Why This Content Is Needed
        expected_headers = [
            'Suggested Topic', 'Primary Keyword', 'Secondary Keywords', 'Total Impressions',
            'Avg Position', 'Cannibalization Risk', 'Why This Content Is Needed', 'Suggested Page Type', 'Priority Score'
        ]

        # Check if sheet has data or the "no opportunities" message
        first_cell = ws.cell(row=1, column=1).value
        if first_cell == 'No new content topic opportunities found':
            print(f"  [OK] Sheet has 'no opportunities' message (data-dependent)")
        else:
            # Check headers
            for col_idx, expected in enumerate(expected_headers, start=1):
                actual = ws.cell(row=1, column=col_idx).value
                if actual != expected:
                    errors.append(f"  {sheet_name}: Column {col_idx} header - expected '{expected}', got '{actual}'")
                else:
                    print(f"  [OK] Column {col_idx}: '{expected}'")

            # Check freeze panes
            if ws.freeze_panes != 'A2':
                warnings.append(f"  {sheet_name}: Freeze panes expected 'A2', got '{ws.freeze_panes}'")
            else:
                print(f"  [OK] Freeze panes at A2")

            # Count data rows (topic-based sheet, capped at 20)
            row_count = ws.max_row - 1  # Exclude header
            print(f"  [OK] Data rows: {row_count}")

            # Verify Top 20 cap
            if row_count > 20:
                errors.append(f"  {sheet_name}: Expected max 20 topics, got {row_count}")
            else:
                print(f"  [OK] Row count within Top 20 cap")

            # Verify no URL column exists (should be topic-based, not URL-based)
            for col_idx in range(1, ws.max_column + 1):
                header = ws.cell(row=1, column=col_idx).value
                if header and 'URL' in str(header).upper():
                    errors.append(f"  {sheet_name}: Found URL column '{header}' - sheet should be topic-based, not URL-based")

    print()

    # Close workbook
    wb.close()

    # ==========================================================================
    # Report Results
    # ==========================================================================
    print("=" * 60)

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  [FAIL] {error}")
        print()

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  [WARN] {warning}")
        print()

    if not errors:
        print("[OK] All verifications PASSED")
        return True
    else:
        print(f"[FAIL] {len(errors)} verification(s) FAILED")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_analytical_sheets.py <path_to_wqa_output.xlsx>")
        print()
        print("Example:")
        print("  python test_analytical_sheets.py wqa_report.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    success = verify_analytical_sheets(excel_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
