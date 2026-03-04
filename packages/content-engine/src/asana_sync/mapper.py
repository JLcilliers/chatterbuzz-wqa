"""
Asana Field Mapper
==================
Maps WQA action types and priorities to Asana project sections and
custom-field values.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Action -> Section mapping
# ---------------------------------------------------------------------------

ACTION_TO_SECTION: dict[str, str] = {
    "Redirect": "Redirects",
    "Add Schema": "Schema",
    "Content Rewrite": "Content Rewrites",
    "Optimize Meta": "Meta Optimization",
    "Internal Linking": "Internal Links",
    "Thin Content": "Content Expansion",
    "De-indexed Alert": "Alerts",
    "Crawl Budget Alert": "Alerts",
    "Traffic Anomaly": "Alerts",
}


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------

_PRIORITY_MAP: dict[str, str] = {
    "critical": "P0",
    "high": "P1",
    "medium": "P2",
    "low": "P3",
}


def map_priority(wqa_priority: str) -> str:
    """Convert a WQA priority label to an Asana custom-field value.

    Args:
        wqa_priority: One of ``critical``, ``high``, ``medium``, ``low``
            (case-insensitive).

    Returns:
        Asana priority string (``P0`` -- ``P3``).  Falls back to ``P2`` for
        unrecognised values.
    """
    return _PRIORITY_MAP.get(wqa_priority.lower().strip(), "P2")
