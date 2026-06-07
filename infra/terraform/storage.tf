// The shared `garassino-ml-artifacts` bucket already exists for the workspace.
// We model it here so we can attach IAM bindings scoped to the deepmaze/ prefix,
// but it must be `terraform import`ed before the first `terraform apply` and is
// protected from destruction so `terraform destroy` doesn't tear it down.
//
//   terraform import google_storage_bucket.artifacts garassino-ml-artifacts

resource "google_storage_bucket" "artifacts" {
  name     = var.assets_bucket
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle {
    prevent_destroy = true
  }
}
