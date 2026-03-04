"""
Pipeline Orchestrator
=====================
Runs data-pipeline workers in dependency order, tracks execution state,
and logs progress to the pipeline_runs table in Supabase.

Worker dependency graph (current):
  crawl_worker   (no deps -- file upload)
  ga4_worker     (no deps -- API fetch)
  gsc_worker     (no deps -- API fetch)
  backlink_worker(no deps -- file upload or API)
  gbp_worker     (no deps -- API fetch, stub)

All workers are independent today; the orchestrator still enforces the
ordering so that future workers that *do* have dependencies (e.g. an
analysis worker that needs crawl + GA4 + GSC) slot in naturally.
"""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    PARTIAL = 'partial'       # Some workers succeeded, some failed/skipped
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class WorkerStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    SKIPPED = 'skipped'
    FAILED = 'failed'


# ---------------------------------------------------------------------------
# Worker registry
# ---------------------------------------------------------------------------

class WorkerSpec:
    """Describes a single worker that the orchestrator can invoke."""

    def __init__(
        self,
        name: str,
        run_fn: Callable[..., dict],
        depends_on: Optional[List[str]] = None,
    ):
        self.name = name
        self.run_fn = run_fn
        self.depends_on = depends_on or []


# Default worker ordering (importable functions resolved at registration time)
_WORKER_REGISTRY: Dict[str, WorkerSpec] = {}


def register_worker(
    name: str,
    run_fn: Callable[..., dict],
    depends_on: Optional[List[str]] = None,
):
    """Register a worker function with the orchestrator.

    Args:
        name: Unique worker name (e.g. 'ga4', 'gsc', 'crawl').
        run_fn: Callable that accepts (supabase_client, client_id, **kwargs)
                and returns a dict with at least 'status' and 'error' keys.
        depends_on: List of worker names that must complete successfully
                    before this worker can start.
    """
    _WORKER_REGISTRY[name] = WorkerSpec(name=name, run_fn=run_fn, depends_on=depends_on)
    logger.debug(f"Registered pipeline worker: {name}")


def _auto_register_default_workers():
    """Lazy-import and register the built-in workers the first time the
    orchestrator is used, if they have not been registered manually."""
    if _WORKER_REGISTRY:
        return  # already populated

    try:
        from .workers.ga4_worker import run_ga4_worker
        register_worker('ga4', run_ga4_worker)
    except ImportError:
        logger.debug("ga4_worker not available for auto-registration")

    try:
        from .workers.gsc_worker import run_gsc_worker
        register_worker('gsc', run_gsc_worker)
    except ImportError:
        logger.debug("gsc_worker not available for auto-registration")

    try:
        from .workers.crawl_worker import run_crawl_worker
        register_worker('crawl', run_crawl_worker)
    except ImportError:
        logger.debug("crawl_worker not available for auto-registration")

    try:
        from .workers.backlink_worker import run_backlink_worker
        register_worker('backlink', run_backlink_worker)
    except ImportError:
        logger.debug("backlink_worker not available for auto-registration")

    try:
        from .workers.gbp_worker import run_gbp_worker
        register_worker('gbp', run_gbp_worker)
    except ImportError:
        logger.debug("gbp_worker not available for auto-registration")


# ---------------------------------------------------------------------------
# Supabase pipeline_runs helpers
# ---------------------------------------------------------------------------

def _create_pipeline_run(supabase_client, client_id: str, workers: List[str]) -> Optional[str]:
    """Insert a new pipeline_runs row and return its ID."""
    try:
        record = {
            'client_id': client_id,
            'status': PipelineStatus.RUNNING.value,
            'workers_requested': workers,
            'started_at': datetime.utcnow().isoformat(),
            'worker_results': {},
        }
        result = supabase_client.table('pipeline_runs').insert(record).execute()
        run_id = result.data[0]['id'] if result.data else None
        logger.info(f"[Orchestrator] Created pipeline_run {run_id} for client={client_id}")
        return run_id
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to create pipeline_run: {e}")
        return None


def update_pipeline_status(
    supabase_client,
    run_id: str,
    status: PipelineStatus,
    worker_results: Optional[Dict] = None,
    error: Optional[str] = None,
):
    """Update an existing pipeline_runs row with current status and results."""
    if not run_id:
        return

    try:
        update_data: Dict[str, Any] = {
            'status': status.value,
            'updated_at': datetime.utcnow().isoformat(),
        }
        if worker_results is not None:
            update_data['worker_results'] = worker_results
        if error is not None:
            update_data['error'] = error
        if status in (PipelineStatus.SUCCESS, PipelineStatus.PARTIAL, PipelineStatus.FAILED):
            update_data['completed_at'] = datetime.utcnow().isoformat()

        supabase_client.table('pipeline_runs').update(update_data).eq('id', run_id).execute()
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to update pipeline_run {run_id}: {e}")


# ---------------------------------------------------------------------------
# Single-worker execution
# ---------------------------------------------------------------------------

def run_worker(
    supabase_client,
    client_id: str,
    worker_name: str,
    **kwargs,
) -> dict:
    """Execute a single named worker.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        worker_name: Name of the registered worker to execute.
        **kwargs: Additional keyword arguments forwarded to the worker function.

    Returns:
        dict with keys: status, error, and any worker-specific data.
    """
    _auto_register_default_workers()

    spec = _WORKER_REGISTRY.get(worker_name)
    if spec is None:
        return {'status': 'error', 'error': f"Unknown worker: {worker_name}"}

    logger.info(f"[Orchestrator] Running worker '{worker_name}' for client={client_id}")
    try:
        result = spec.run_fn(supabase_client, client_id, **kwargs)
        return result
    except Exception as e:
        logger.error(f"[Orchestrator] Worker '{worker_name}' raised: {e}")
        return {'status': 'error', 'error': str(e)}


# ---------------------------------------------------------------------------
# Full pipeline execution
# ---------------------------------------------------------------------------

def _resolve_execution_order(worker_names: List[str]) -> List[str]:
    """Topological sort of requested workers based on depends_on.

    Workers whose dependencies are not in the requested set are placed
    first (their deps are assumed already satisfied).
    """
    # Simple Kahn's algorithm
    in_degree: Dict[str, int] = {name: 0 for name in worker_names}
    adj: Dict[str, List[str]] = {name: [] for name in worker_names}
    requested_set = set(worker_names)

    for name in worker_names:
        spec = _WORKER_REGISTRY.get(name)
        if spec:
            for dep in spec.depends_on:
                if dep in requested_set:
                    adj[dep].append(name)
                    in_degree[name] += 1

    queue = [n for n in worker_names if in_degree[n] == 0]
    order: List[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbour in adj.get(node, []):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    # If there are remaining nodes not in order, append them (cycle fallback)
    for name in worker_names:
        if name not in order:
            order.append(name)

    return order


def run_full_pipeline(
    supabase_client,
    client_id: str,
    workers: Optional[List[str]] = None,
    worker_kwargs: Optional[Dict[str, dict]] = None,
) -> dict:
    """Execute the full data pipeline for a client.

    Runs the requested workers in dependency order, logs progress to
    the pipeline_runs table, and returns a summary.

    Args:
        supabase_client: Initialised Supabase client.
        client_id: Unique identifier for the client / project.
        workers: List of worker names to execute. If None, runs all
                 registered workers.
        worker_kwargs: Optional mapping of worker_name -> kwargs dict
                       to pass to each worker's run function.

    Returns:
        dict with keys: run_id, status, worker_results, error.
    """
    _auto_register_default_workers()

    if workers is None:
        workers = list(_WORKER_REGISTRY.keys())

    worker_kwargs = worker_kwargs or {}

    # Create pipeline run record
    run_id = _create_pipeline_run(supabase_client, client_id, workers)

    # Determine execution order
    execution_order = _resolve_execution_order(workers)
    logger.info(f"[Orchestrator] Execution order: {execution_order}")

    worker_results: Dict[str, dict] = {}
    succeeded = 0
    failed = 0
    skipped = 0

    for worker_name in execution_order:
        spec = _WORKER_REGISTRY.get(worker_name)
        if spec is None:
            worker_results[worker_name] = {'status': WorkerStatus.FAILED.value, 'error': 'Not registered'}
            failed += 1
            continue

        # Check dependencies
        deps_met = True
        for dep in spec.depends_on:
            dep_result = worker_results.get(dep, {})
            if dep_result.get('status') != WorkerStatus.SUCCESS.value:
                deps_met = False
                break

        if not deps_met:
            logger.warning(f"[Orchestrator] Skipping '{worker_name}' -- dependencies not met")
            worker_results[worker_name] = {
                'status': WorkerStatus.SKIPPED.value,
                'error': 'Dependency not met',
            }
            skipped += 1
            update_pipeline_status(supabase_client, run_id, PipelineStatus.RUNNING, worker_results)
            continue

        # Execute
        logger.info(f"[Orchestrator] Executing worker '{worker_name}'")
        try:
            kwargs = worker_kwargs.get(worker_name, {})
            result = spec.run_fn(supabase_client, client_id, **kwargs)
            status_val = result.get('status', 'error')

            if status_val in ('success', 'empty'):
                worker_results[worker_name] = {
                    'status': WorkerStatus.SUCCESS.value,
                    **result,
                }
                succeeded += 1
            else:
                worker_results[worker_name] = {
                    'status': WorkerStatus.FAILED.value,
                    **result,
                }
                failed += 1

        except Exception as e:
            logger.error(f"[Orchestrator] Worker '{worker_name}' exception: {e}\n{traceback.format_exc()}")
            worker_results[worker_name] = {
                'status': WorkerStatus.FAILED.value,
                'error': str(e),
            }
            failed += 1

        # Update pipeline run after each worker
        update_pipeline_status(supabase_client, run_id, PipelineStatus.RUNNING, worker_results)

    # Determine final pipeline status
    if failed == 0 and skipped == 0:
        final_status = PipelineStatus.SUCCESS
    elif succeeded > 0:
        final_status = PipelineStatus.PARTIAL
    else:
        final_status = PipelineStatus.FAILED

    update_pipeline_status(supabase_client, run_id, final_status, worker_results)

    summary = {
        'run_id': run_id,
        'status': final_status.value,
        'worker_results': worker_results,
        'succeeded': succeeded,
        'failed': failed,
        'skipped': skipped,
        'error': None,
    }

    logger.info(
        f"[Orchestrator] Pipeline complete: status={final_status.value}, "
        f"succeeded={succeeded}, failed={failed}, skipped={skipped}"
    )

    return summary
