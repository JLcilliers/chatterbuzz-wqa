"""
Asana Sync
==========
Creates and manages Asana tasks that mirror WQA results and monitoring
alerts.  Uses the Asana REST API directly via ``httpx``.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from .mapper import ACTION_TO_SECTION, map_priority

logger = logging.getLogger(__name__)

ASANA_API_BASE = "https://app.asana.com/api/1.0"


class AsanaSync:
    """Synchronises WQA outputs and alerts to an Asana project.

    Args:
        pat: Asana Personal Access Token.
        workspace_id: Asana workspace GID.
        project_id: Asana project GID where tasks will be created.
    """

    def __init__(self, pat: str, workspace_id: str, project_id: str) -> None:
        self.pat = pat
        self.workspace_id = workspace_id
        self.project_id = project_id
        self._headers = {
            "Authorization": f"Bearer {pat}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Cache: section name -> section GID
        self._section_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_section(self, section_name: str) -> Optional[str]:
        """Return the GID of *section_name*, creating it if necessary."""
        if section_name in self._section_cache:
            return self._section_cache[section_name]

        # List existing sections
        url = f"{ASANA_API_BASE}/projects/{self.project_id}/sections"
        try:
            resp = httpx.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            sections = resp.json().get("data", [])
            for s in sections:
                self._section_cache[s["name"]] = s["gid"]

            if section_name in self._section_cache:
                return self._section_cache[section_name]

            # Create new section
            create_resp = httpx.post(
                url,
                headers=self._headers,
                json={"data": {"name": section_name}},
                timeout=30,
            )
            create_resp.raise_for_status()
            gid = create_resp.json()["data"]["gid"]
            self._section_cache[section_name] = gid
            logger.info(f"[Asana] Created section '{section_name}' ({gid})")
            return gid

        except Exception as e:
            logger.error(f"[Asana] Section lookup/create failed for '{section_name}': {e}")
            return None

    def _find_existing_task(self, source_type: str, source_id: str) -> Optional[str]:
        """Search for a task whose notes contain the dedup marker.

        Returns the task GID if found, else ``None``.
        """
        dedup_marker = f"[wqa:{source_type}:{source_id}]"
        search_url = (
            f"{ASANA_API_BASE}/workspaces/{self.workspace_id}/tasks/search"
        )
        params = {
            "projects.any": self.project_id,
            "text": dedup_marker,
            "opt_fields": "gid",
        }

        try:
            resp = httpx.get(
                search_url, headers=self._headers, params=params, timeout=30,
            )
            resp.raise_for_status()
            tasks = resp.json().get("data", [])
            if tasks:
                return tasks[0]["gid"]
        except Exception as e:
            logger.warning(f"[Asana] Dedup search failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_task(
        self,
        name: str,
        section: str,
        notes: str,
        priority: str,
        client_id: str,
        source_type: str,
        source_id: str,
    ) -> Optional[str]:
        """Create an Asana task, deduplicating by *source_type* + *source_id*.

        If a task already exists with the same dedup marker the method returns
        the existing task GID without creating a duplicate.

        Args:
            name: Task title.
            section: Asana section name (looked up / created automatically).
            notes: Task description / body.
            priority: Asana priority value (e.g. ``P0``).
            client_id: Client identifier (included in notes for traceability).
            source_type: Origin category (``wqa``, ``deindex``, ``crawl``, ``anomaly``).
            source_id: Unique ID of the source record.

        Returns:
            The task GID on success, or ``None`` on failure.
        """
        # Deduplication check
        existing_gid = self._find_existing_task(source_type, source_id)
        if existing_gid:
            logger.info(
                f"[Asana] Task already exists ({existing_gid}) for "
                f"{source_type}:{source_id} -- skipping."
            )
            return existing_gid

        dedup_marker = f"[wqa:{source_type}:{source_id}]"
        full_notes = f"{notes}\n\n---\nclient: {client_id}\n{dedup_marker}"

        section_gid = self._get_or_create_section(section)

        task_data: dict = {
            "data": {
                "name": name,
                "notes": full_notes,
                "projects": [self.project_id],
                "workspace": self.workspace_id,
            }
        }

        # Add to section via memberships
        if section_gid:
            task_data["data"]["memberships"] = [
                {"project": self.project_id, "section": section_gid}
            ]

        try:
            resp = httpx.post(
                f"{ASANA_API_BASE}/tasks",
                headers=self._headers,
                json=task_data,
                timeout=30,
            )
            resp.raise_for_status()
            task_gid = resp.json()["data"]["gid"]
            logger.info(f"[Asana] Created task '{name}' ({task_gid})")
            return task_gid

        except Exception as e:
            logger.error(f"[Asana] Task creation failed for '{name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Bulk sync helpers
    # ------------------------------------------------------------------

    def sync_wqa_results(self, client_id: str, supabase_client) -> list[dict]:
        """Read pending WQA results and create corresponding Asana tasks.

        Reads rows from the ``wqa_results`` table where
        ``asana_synced = false``, creates tasks, and updates both the
        ``wqa_results`` row and the ``asana_tasks`` tracking table.

        Args:
            client_id: Client / project identifier.
            supabase_client: Initialised Supabase client.

        Returns:
            List of dicts with keys: wqa_id, task_gid, status.
        """
        synced: list[dict] = []

        try:
            response = (
                supabase_client.table("wqa_results")
                .select("*")
                .eq("client_id", client_id)
                .eq("asana_synced", False)
                .execute()
            )
            rows = response.data or []
        except Exception as e:
            logger.error(f"[Asana] Failed to read wqa_results: {e}")
            return synced

        for row in rows:
            wqa_id = str(row.get("id", ""))
            action = row.get("action", "")
            url = row.get("url", "")
            priority_raw = row.get("priority", "medium")
            notes = row.get("recommendation", "")

            section = ACTION_TO_SECTION.get(action, "Uncategorised")
            priority = map_priority(priority_raw)
            task_name = f"[{priority}] {action}: {url}"

            task_gid = self.create_task(
                name=task_name,
                section=section,
                notes=notes,
                priority=priority,
                client_id=client_id,
                source_type="wqa",
                source_id=wqa_id,
            )

            status = "created" if task_gid else "failed"
            synced.append({"wqa_id": wqa_id, "task_gid": task_gid, "status": status})

            if task_gid:
                now_iso = datetime.utcnow().isoformat()
                try:
                    # Mark WQA row as synced
                    supabase_client.table("wqa_results").update(
                        {"asana_synced": True}
                    ).eq("id", wqa_id).execute()

                    # Write tracking record
                    supabase_client.table("asana_tasks").upsert(
                        {
                            "client_id": client_id,
                            "task_gid": task_gid,
                            "source_type": "wqa",
                            "source_id": wqa_id,
                            "section": section,
                            "priority": priority,
                            "synced_at": now_iso,
                        },
                        on_conflict="client_id,task_gid",
                    ).execute()
                except Exception as e:
                    logger.error(
                        f"[Asana] Failed to update Supabase after task creation: {e}"
                    )

        logger.info(
            f"[Asana] sync_wqa_results complete: {len(synced)} tasks processed "
            f"for client={client_id}"
        )
        return synced

    def sync_alerts(
        self,
        alerts: list[dict],
        client_id: str,
        supabase_client,
    ) -> list[dict]:
        """Create Asana tasks for alerts from the index monitor or anomaly detection.

        Each alert dict must contain at minimum: ``url``, ``alert_type``
        (one of ``De-indexed Alert``, ``Crawl Budget Alert``,
        ``Traffic Anomaly``), and ``detail`` (description text).  An
        optional ``priority`` key defaults to ``high``.

        Args:
            alerts: List of alert dicts.
            client_id: Client / project identifier.
            supabase_client: Initialised Supabase client.

        Returns:
            List of dicts with keys: alert_url, task_gid, status.
        """
        synced: list[dict] = []

        for alert in alerts:
            url = alert.get("url", "unknown")
            alert_type = alert.get("alert_type", "Traffic Anomaly")
            detail = alert.get("detail", "")
            priority_raw = alert.get("priority", "high")

            section = ACTION_TO_SECTION.get(alert_type, "Alerts")
            priority = map_priority(priority_raw)
            source_id = f"{alert_type}:{url}"
            task_name = f"[{priority}] {alert_type}: {url}"

            task_gid = self.create_task(
                name=task_name,
                section=section,
                notes=detail,
                priority=priority,
                client_id=client_id,
                source_type="alert",
                source_id=source_id,
            )

            status = "created" if task_gid else "failed"
            synced.append({"alert_url": url, "task_gid": task_gid, "status": status})

            if task_gid:
                now_iso = datetime.utcnow().isoformat()
                try:
                    supabase_client.table("asana_tasks").upsert(
                        {
                            "client_id": client_id,
                            "task_gid": task_gid,
                            "source_type": "alert",
                            "source_id": source_id,
                            "section": section,
                            "priority": priority,
                            "synced_at": now_iso,
                        },
                        on_conflict="client_id,task_gid",
                    ).execute()
                except Exception as e:
                    logger.error(
                        f"[Asana] Failed to write asana_tasks record: {e}"
                    )

        logger.info(
            f"[Asana] sync_alerts complete: {len(synced)} alerts processed "
            f"for client={client_id}"
        )
        return synced
