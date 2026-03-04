"""
Phase 7 — Rank Math & Yoast CSV Exporters

Produce CSV strings ready for bulk import into Rank Math SEO or Yoast SEO
WordPress plugins.
"""

import csv
import io
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank Math export
# ---------------------------------------------------------------------------

def export_for_rank_math(schemas: List[dict]) -> str:
    """Generate a CSV string for Rank Math bulk schema import.

    Rank Math expects a CSV with at least two columns:

    - ``url`` — the page URL
    - ``schema_json`` — a JSON-LD string (double-quote escaped so it survives
      CSV parsing)

    Parameters
    ----------
    schemas : list[dict]
        Output from ``bulk_generate`` — each dict must contain ``url`` and
        ``schema_json`` keys.  Entries with an ``error`` key are skipped.

    Returns
    -------
    str
        CSV content as a string.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["url", "schema_json"])

    exported = 0
    for entry in schemas:
        if "error" in entry:
            logger.debug("Skipping errored entry: %s", entry.get("url", "?"))
            continue

        url = entry.get("url", "")
        schema_obj = entry.get("schema_json", {})

        # Wrap in a <script> tag representation that Rank Math recognises.
        schema_str = json.dumps(schema_obj, ensure_ascii=False)

        writer.writerow([url, schema_str])
        exported += 1

    logger.info("Rank Math CSV: exported %d schemas.", exported)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Yoast export
# ---------------------------------------------------------------------------

def export_for_yoast(schemas: List[dict]) -> str:
    """Generate a CSV string for Yoast SEO bulk schema import.

    Yoast uses a slightly different format:

    - ``url`` — the page URL
    - ``schema_type`` — the ``@type`` value (e.g. ``Article``, ``FAQPage``)
    - ``schema_graph`` — the full JSON-LD string representing the schema graph
      piece to merge into Yoast's ``@graph`` array

    Parameters
    ----------
    schemas : list[dict]
        Output from ``bulk_generate``.

    Returns
    -------
    str
        CSV content as a string.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["url", "schema_type", "schema_graph"])

    exported = 0
    for entry in schemas:
        if "error" in entry:
            logger.debug("Skipping errored entry: %s", entry.get("url", "?"))
            continue

        url = entry.get("url", "")
        schema_obj = entry.get("schema_json", {})
        schema_type = schema_obj.get("@type", "WebPage")

        # Yoast graph piece: strip the @context (Yoast adds its own wrapper)
        # and add an @id for graph stitching.
        graph_piece = {k: v for k, v in schema_obj.items() if k != "@context"}
        graph_piece.setdefault("@id", f"{url}#/schema/{schema_type.lower()}")

        schema_graph_str = json.dumps(graph_piece, ensure_ascii=False)

        writer.writerow([url, schema_type, schema_graph_str])
        exported += 1

    logger.info("Yoast CSV: exported %d schemas.", exported)
    return buf.getvalue()
