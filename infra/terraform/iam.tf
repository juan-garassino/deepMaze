// Runtime SA used by the Cloud Run service to read the deepmaze/ prefix of the
// shared bucket and write Cloud Trace spans.

resource "google_service_account" "backend" {
  account_id   = "deepmaze-backend"
  display_name = "deepMaze backend runtime"
  description  = "Cloud Run runtime SA — sync_assets.py reads gs://${var.assets_bucket}/${var.assets_prefix}, OTEL writes to Cloud Trace."
}

// Read-only on the deepmaze/ prefix only (not the whole shared bucket).
resource "google_storage_bucket_iam_member" "backend_reader" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.backend.email}"

  condition {
    title       = "deepmaze-prefix-only"
    description = "Limit reads to gs://${var.assets_bucket}/${var.assets_prefix}*"
    // storage.objects.list is evaluated against the BUCKET resource, which
    // never matches the objects/ prefix — without the second clause,
    // sync_assets.py's list_blobs(prefix=...) gets a 403 and the service
    // boots with zero models.
    expression = "resource.name.startsWith(\"projects/_/buckets/${var.assets_bucket}/objects/${var.assets_prefix}\") || api.getAttribute(\"storage.googleapis.com/objectListPrefix\", \"\").startsWith(\"${var.assets_prefix}\")"
  }
}

resource "google_project_iam_member" "backend_trace" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

// Allow GitHub Actions (running on this repo, via garassino-op's WIF pool) to
// impersonate the backend SA. The principalSet:// member binds at the POOL
// (not provider) level — the provider's own attribute_condition is what
// actually scopes which repos can mint tokens.
// deploy.yml authenticates AS this SA (via WIF) and then runs
// `gcloud run services replace` + `add-iam-policy-binding`: that needs
// run.admin on the project and actAs on the runtime SA (an SA does not
// implicitly have actAs on itself).
resource "google_project_iam_member" "backend_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_service_account_iam_member" "backend_self_actas" {
  service_account_id = google_service_account.backend.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_service_account_iam_member" "wif_impersonation" {
  service_account_id = google_service_account.backend.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${var.wif_pool_id}/attribute.repository/${var.github_repo}"
}
