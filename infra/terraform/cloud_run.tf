// Cloud Run v2 service for the inference backend.
// The image LIVES on GHCR (workspace policy: deep-* on GHCR, not GAR), but
// Cloud Run cannot pull ghcr.io directly — it only accepts Artifact
// Registry / gcr.io / Docker Hub. We pull through an Artifact Registry
// REMOTE repository that proxies ghcr.io (one-time setup, see README):
//   gcloud artifacts repositories create ghcr-remote \
//     --repository-format=docker --mode=remote-repository \
//     --remote-docker-repo=https://ghcr.io \
//     --location=<region> --project=<project>
// Public GHCR package ⇒ no upstream credentials needed on the remote repo.

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
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ghcr_remote_repo}/${var.ghcr_owner}/${var.image_name}:${var.image_tag}"

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

  // deploy.yml replaces the service with a new image on every merge; TF owns
  // the infrastructure, CI owns the image. Without this, the next apply
  // would roll the live revision back to var.image_tag.
  lifecycle {
    ignore_changes = [template[0].containers[0].image]
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
