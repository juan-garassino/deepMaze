# Observability (F)

Subsystem F of the MLOps loop. See [`architecture.md`](architecture.md) for how it fits into the broader flow.

Five surfaces, increasing in operational distance from the code:

| Surface | What you see | Setup |
|---|---|---|
| MLflow | Experiment runs, metrics curves, params, artifacts, model registry | `infra/mlflow/` — local or Cloud Run |
| Prefect | Flow runs, retries, schedules, parameter history | Prefect Cloud free tier, or `infra/prefect/docker-compose.yml` |
| Cloud Run metrics | Request count, latency, instance count, cold starts | Auto — `Cloud Run → deepmaze-backend → metrics` tab |
| Cloud Trace | Per-request spans from FastAPI, with route + status | OTEL in `web/otel.py`, enabled by `OTEL_TRACES_EXPORTER=gcp_trace` |
| Cloud Logging | Structured logs from gunicorn + app | Auto on Cloud Run; query with `resource.type="cloud_run_revision"` |

## OTEL config (Cloud Run env vars)

| Var | Example | Honored by |
|---|---|---|
| `OTEL_TRACES_EXPORTER` | `gcp_trace` · `console` | `web/otel.py::instrument` (gates everything) |
| `OTEL_SERVICE_NAME` | `deepmaze-backend` | `web/otel.py` (defaults to `deepmaze-backend`) |
| `OTEL_SERVICE_VERSION` | `2026-06-03` | `web/otel.py` (defaults to `dev`) |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` · `always_on` · `traceidratio` · `always_off` | `web/otel.py::_sampler_from_env` |
| `OTEL_TRACES_SAMPLER_ARG` | `0.1` (10 % sampled) | `web/otel.py::_sampler_from_env` |
| `GCP_PROJECT_ID` | `your-project` | `web/otel.py` (passed to `CloudTraceSpanExporter`) |

Local debug: set `OTEL_TRACES_EXPORTER=console` to dump spans to stdout.

## Things deferred (out of scope)

- Grafana on Cloud Run reading Cloud Monitoring + MLflow Postgres — possible but not wired.
- Cost dashboards / budget alerts.
- LangSmith-style request log (no LLM in this project; the FastAPI route logs are enough).
