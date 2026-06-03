"""Prefect flow: trigger external retraining → wait → fetch best MLflow run.

External training runs in Colab (manual today; Vertex AI a later upgrade).
This flow watches MLflow for a new run under EXPERIMENT_NAME that finishes
with eval_success_rate above the current registered champion, then delegates
to promote_flow to commit + redeploy.
"""

from __future__ import annotations

import os
import time

import mlflow
from prefect import flow, get_run_logger, task

from flows.promote_flow import promote_flow

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "deepmaze")
POLL_SECONDS = int(os.environ.get("RETRAIN_POLL_SECONDS", "60"))
MAX_WAIT_MIN = int(os.environ.get("RETRAIN_MAX_WAIT_MIN", "180"))
EVAL_METRIC = os.environ.get("RETRAIN_EVAL_METRIC", "eval_success_rate")


@task
def find_champion(client: mlflow.MlflowClient) -> float:
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
        order_by=[f"metrics.{EVAL_METRIC} DESC"],
        max_results=1,
    )
    return runs[0].data.metrics.get(EVAL_METRIC, float("-inf")) if runs else float("-inf")


@task
def wait_for_new_run(client: mlflow.MlflowClient, since_ts_ms: int) -> str:
    log = get_run_logger()
    deadline = time.time() + MAX_WAIT_MIN * 60
    exp_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    while time.time() < deadline:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"attributes.status = 'FINISHED' and attributes.start_time > {since_ts_ms}",
            order_by=["attributes.end_time DESC"],
            max_results=1,
        )
        if runs:
            log.info(f"new finished run: {runs[0].info.run_id}")
            return runs[0].info.run_id
        log.info(f"no new run yet; sleeping {POLL_SECONDS}s")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"no new run finished within {MAX_WAIT_MIN} min")


@flow(name="retrain_flow")
def retrain_flow(open_pr: bool = True) -> str | None:
    log = get_run_logger()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set")
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()

    champion = find_champion(client)
    log.info(f"current champion {EVAL_METRIC} = {champion:.4f}")

    since_ms = int(time.time() * 1000)
    new_run_id = wait_for_new_run(client, since_ms)
    new_metric = client.get_run(new_run_id).data.metrics.get(EVAL_METRIC, float("-inf"))
    log.info(f"new run {EVAL_METRIC} = {new_metric:.4f}")

    if new_metric <= champion:
        log.info("new run did not beat champion; skipping promotion")
        return None

    return promote_flow(run_id=new_run_id, open_pr=open_pr)


if __name__ == "__main__":
    retrain_flow()
