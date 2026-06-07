// deepMaze — minimal show-and-destroy IaC targeting garassino-ml / europe-west1.
//
// Per workspace policy (root CLAUDE.md § "GCP architecture"):
//   - State lives in garassino-op's bucket: gs://garassino-op-tf-state/deepmaze/
//   - WIF pool lives in garassino-op — we only bind to it from here
//   - Bucket garassino-ml-artifacts is shared across the deep-* family;
//     we import it (do not create) so `terraform destroy` leaves it alone.
//
// Usage:
//   terraform init
//   terraform import google_storage_bucket.artifacts garassino-ml-artifacts
//   terraform apply  -var "image_tag=$(git rev-parse --short HEAD)"
//   terraform destroy  # tears down deepMaze-only resources; bucket survives

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  backend "gcs" {
    bucket = "garassino-op-tf-state"
    prefix = "deepmaze"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
