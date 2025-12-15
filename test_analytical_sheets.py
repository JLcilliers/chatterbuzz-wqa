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

        # Check for expected headers with Content Action Recommendation columns
        expected_headers = [
            'Suggested Topic', 'Primary Keyword', 'Secondary Keywords', 'Total Impressions',
            'Avg Position', 'Recommended Action', 'Primary URL', 'Secondary URLs', 'Reasoning',
            'Suggested Page Type', 'Priority Score'
        ]

        # Valid action values
        valid_actions = ['Create new page', 'Expand existing page', 'Consolidate pages']

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

            # ======================================================================
            # Content Action Recommendation Validation
            # ======================================================================
            print(f"  Validating Content Action Recommendations...")

            # Column indices (1-based)
            action_col = 6      # Recommended Action
            primary_url_col = 7  # Primary URL
            secondary_urls_col = 8  # Secondary URLs

            action_counts = {'Create new page': 0, 'Expand existing page': 0, 'Consolidate pages': 0}
            rows_without_action = 0
            invalid_actions = []
            create_with_url = 0
            consolidate_without_multiple = 0

            for row_idx in range(2, ws.max_row + 1):
                action = ws.cell(row=row_idx, column=action_col).value
                primary_url = ws.cell(row=row_idx, column=primary_url_col).value or ''
                secondary_urls = ws.cell(row=row_idx, column=secondary_urls_col).value or ''

                # Check each topic has exactly one recommendation
                if not action:
                    rows_without_action += 1
                elif action not in valid_actions:
                    invalid_actions.append(f"Row {row_idx}: '{action}'")
                else:
                    action_counts[action] += 1

                    # "Create new page" rows should have empty Primary URL
                    if action == 'Create new page':
                        if primary_url.strip():
                            create_with_url += 1

                    # "Consolidate pages" rows should list >= 2 URLs in Secondary URLs
                    if action == 'Consolidate pages':
                        url_count = len([u for u in secondary_urls.split(',') if u.strip()]) if secondary_urls else 0
                        if url_count < 2:
                            consolidate_without_multiple += 1

            # Report action distribution
            total_actions = sum(action_counts.values())
            print(f"    Actions: Create={action_counts['Create new page']}, Expand={action_counts['Expand existing page']}, Consolidate={action_counts['Consolidate pages']}")

            # Validate: every topic has exactly one recommendation
            if rows_without_action > 0:
                errors.append(f"  {sheet_name}: {rows_without_action} row(s) missing Recommended Action")
            else:
                print(f"  [OK] Every topic has a Recommended Action")

            # Validate: all actions are valid
            if invalid_actions:
                for inv in invalid_actions[:5]:  # Show first 5
                    errors.append(f"  {sheet_name}: Invalid action - {inv}")
            else:
                print(f"  [OK] All actions are valid values")

            # Validate: "Create new page" rows have no Primary URL
            if create_with_url > 0:
                errors.append(f"  {sheet_name}: {create_with_url} 'Create new page' row(s) have Primary URL (should be empty)")
            elif action_counts['Create new page'] > 0:
                print(f"  [OK] 'Create new page' rows have empty Primary URL")

            # Validate: "Consolidate pages" rows list >= 2 URLs
            if consolidate_without_multiple > 0:
                errors.append(f"  {sheet_name}: {consolidate_without_multiple} 'Consolidate pages' row(s) have < 2 Secondary URLs")
            elif action_counts['Consolidate pages'] > 0:
                print(f"  [OK] 'Consolidate pages' rows list >= 2 Secondary URLs")

    print()

    # ==========================================================================
    # Verify Sheet 4: Redirect & Merge Plan
    # ==========================================================================
    sheet_name = 'Redirect & Merge Plan'
    print(f"Checking '{sheet_name}'...")

    if sheet_name not in wb.sheetnames:
        # Sheet may not exist if no consolidations were found
        print(f"  [INFO] Sheet '{sheet_name}' not found (expected if no consolidations)")
    else:
        ws = wb[sheet_name]
        print(f"  [OK] Sheet exists")

        # Check for expected headers
        expected_headers = [
            'Topic', 'Recommended Action', 'Primary URL', 'Secondary URL',
            'Redirect Type', 'Reason'
        ]

        # Check if sheet has data or the "no redirects" message
        first_cell = ws.cell(row=1, column=1).value
        if first_cell == 'No consolidation redirects needed (no "Consolidate pages" actions found)':
            print(f"  [OK] Sheet has 'no redirects' message (no consolidations found)")
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

            # ======================================================================
            # Redirect & Merge Plan Validation
            # ======================================================================
            print(f"  Validating Redirect Plan...")

            # Column indices (1-based)
            action_col = 2      # Recommended Action
            primary_url_col = 3  # Primary URL
            secondary_url_col = 4  # Secondary URL
            redirect_type_col = 5  # Redirect Type

            secondary_urls_seen = set()
            primary_urls_seen = set()
            non_consolidate_actions = 0
            non_301_types = 0
            duplicate_secondary = 0
            primary_as_secondary = 0

            for row_idx in range(2, ws.max_row + 1):
                action = ws.cell(row=row_idx, column=action_col).value
                primary_url = ws.cell(row=row_idx, column=primary_url_col).value or ''
                secondary_url = ws.cell(row=row_idx, column=secondary_url_col).value or ''
                redirect_type = ws.cell(row=row_idx, column=redirect_type_col).value or ''

                # Check: Only consolidation actions should generate redirects
                if action != 'Consolidate pages':
                    non_consolidate_actions += 1

                # Check: Redirect Type should always be "301"
                if str(redirect_type) != '301':
                    non_301_types += 1

                # Track primary URLs
                if primary_url:
                    primary_urls_seen.add(primary_url)

                # Check: Each Secondary URL appears exactly once
                if secondary_url:
                    if secondary_url in secondary_urls_seen:
                        duplicate_secondary += 1
                    secondary_urls_seen.add(secondary_url)

            # Check: Primary URL never appears as Secondary URL
            for primary in primary_urls_seen:
                if primary in secondary_urls_seen:
                    primary_as_secondary += 1

            # Report validation results
            if non_consolidate_actions > 0:
                errors.append(f"  {sheet_name}: {non_consolidate_actions} row(s) have non-'Consolidate pages' action")
            else:
                print(f"  [OK] All redirects are for 'Consolidate pages' actions")

            if non_301_types > 0:
                errors.append(f"  {sheet_name}: {non_301_types} row(s) have redirect type != '301'")
            else:
                print(f"  [OK] All redirects are type '301'")

            if duplicate_secondary > 0:
                errors.append(f"  {sheet_name}: {duplicate_secondary} duplicate Secondary URL(s) found")
            else:
                print(f"  [OK] Each Secondary URL appears exactly once")

            if primary_as_secondary > 0:
                errors.append(f"  {sheet_name}: {primary_as_secondary} Primary URL(s) also appear as Secondary URL")
            else:
                print(f"  [OK] No Primary URL appears as Secondary URL")

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
