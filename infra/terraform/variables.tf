variable "project_id" {
  description = "GCP project for deepMaze workloads. Workspace policy: garassino-ml."
  type        = string
  default     = "garassino-ml"
}

variable "region" {
  description = "GCP region. Workspace policy: europe-west1 for new resources."
  type        = string
  default     = "europe-west1"
}

variable "image_tag" {
  description = "Tag of the GHCR image to deploy (e.g. a short SHA or 'latest')."
  type        = string
  default     = "latest"
}

variable "ghcr_owner" {
  description = "GitHub owner whose ghcr.io namespace hosts the inference image."
  type        = string
  default     = "juan-garassino"
}

variable "ghcr_remote_repo" {
  description = "Artifact Registry REMOTE repository proxying ghcr.io (created once, out of band — see README)."
  type        = string
  default     = "ghcr-remote"
}

variable "image_name" {
  description = "GHCR repository name for the inference backend."
  type        = string
  default     = "deepmaze-backend"
}

variable "service_name" {
  description = "Cloud Run service name. Must match CLOUD_RUN_SERVICE in deploy.yml."
  type        = string
  default     = "deepmaze-backend"
}

variable "assets_bucket" {
  description = "Shared GCS bucket name for the deep-* family. Imported, not created."
  type        = string
  default     = "garassino-ml-artifacts"
}

variable "assets_prefix" {
  description = "Per-project prefix inside the shared bucket."
  type        = string
  default     = "deepmaze/"
}

variable "cors_origins" {
  description = "Allowed CORS origins for the backend (comma-separated). `*` for demo only."
  type        = string
  default     = "*"
}

// WIF — these reference the existing pool in garassino-op (workspace policy).
// Must be passed in (don't hardcode) so each repo binds its own GitHub repository name.
variable "wif_pool_id" {
  description = "Full resource id of garassino-op's WIF pool (projects/<num>/locations/global/workloadIdentityPools/gh-actions)."
  type        = string
}

variable "github_repo" {
  description = "GitHub <owner>/<repo> string used by the WIF attribute condition."
  type        = string
  default     = "juan-garassino/deepMaze"
}
