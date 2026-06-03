"""Optional OpenTelemetry instrumentation for the FastAPI backend.

Activates only when OTEL_TRACES_EXPORTER is set. Currently supports:
- gcp_trace  → Cloud Trace (via opentelemetry-exporter-gcp-trace)
- console    → stdout, for local debugging

All imports are guarded so the prod image can ship without OTEL deps
if you ever need to slim further.
"""

from __future__ import annotations

import os


def instrument(app) -> None:
    exporter_name = os.environ.get("OTEL_TRACES_EXPORTER", "").lower()
    if not exporter_name:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return  # OTEL deps not installed; silently skip

    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "deepmaze-backend"),
        "service.version": os.environ.get("OTEL_SERVICE_VERSION", "dev"),
    })
    provider = TracerProvider(resource=resource)

    if exporter_name == "gcp_trace":
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        except ImportError:
            return
        project = os.environ.get("GCP_PROJECT_ID")
        provider.add_span_processor(BatchSpanProcessor(
            CloudTraceSpanExporter(project_id=project) if project
            else CloudTraceSpanExporter()
        ))
    elif exporter_name == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        return

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
