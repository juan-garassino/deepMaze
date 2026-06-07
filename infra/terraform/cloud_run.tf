// Cloud Run v2 service for the inference backend.
// Image is pulled from GHCR (workspace policy: deep-* on GHCR, not GAR).
// Public package on GHCR ⇒ no registry-auth needed; if you ever flip the
// package to private, add a secret reference here.

resource "google_cloud_run_v2_service" "backend" {
  name     = var.service_name
  location = var.region

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    service_account = google_service_account.backend.email

    containers {
      image = "ghcr.io/${var.ghcr_owner}/${var.image_name}:${var.image_tag}"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "PORT"
        value = "8000"
      }
      env {
        name  = "CORS_ORIGINS"
        value = var.cors_origins
      }
      env {
        name  = "ASSETS_BUCKET"
        value = var.assets_bucket
      }
      env {
        name  = "ASSETS_PREFIX"
        value = var.assets_prefix
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = "deepmaze-backend"
      }
      env {
        name  = "OTEL_TRACES_EXPORTER"
        value = "gcp_trace"
      }
      env {
        name  = "OTEL_TRACES_SAMPLER"
        value = "parentbased_traceidratio"
      }
      env {
        name  = "OTEL_TRACES_SAMPLER_ARG"
        value = "0.1"
      }

      startup_probe {
        http_get {
          path = "/api/health"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 12
      }
    }

    timeout                          = "300s"
    max_instance_request_concurrency = 20
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

// Public invoker — same posture as the rendered service.yaml under deploy.yml.
// Out of scope per spec: tighten with IAP before any real production traffic.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  name     = google_cloud_run_v2_service.backend.name
  location = google_cloud_run_v2_service.backend.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
