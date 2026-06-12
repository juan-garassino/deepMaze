#!/usr/bin/env bash
# Set the GitHub repo secrets + variables for deepMaze under the post-2026-06-07
# garassino-ml architecture. Idempotent — safe to re-run.
#
# Known constants are written directly. User-supplied values can be passed
# via env vars (no prompt) or, if missing, prompted interactively. Skip a
# prompt with empty input.
#
# Usage:
#   bash scripts/setup-gh-secrets.sh
#   REPO=juan-garassino/deepMaze WIF_PROVIDER="..." bash scripts/setup-gh-secrets.sh
#
# Requires: gh (Logged in via `gh auth status`) + access to the repo.

set -euo pipefail

REPO="${REPO:-juan-garassino/deepMaze}"
echo "Setting GH secrets + variables on: ${REPO}"
echo

# ---------------------------------------------------------------------------
# Known constants — always written.
# ---------------------------------------------------------------------------

echo "Known constants:"
gh variable set GCP_REGION    -R "${REPO}" -b "europe-west1"             && echo "  ✓ variable GCP_REGION = europe-west1"
gh variable set ASSETS_BUCKET -R "${REPO}" -b "garassino-ml-artifacts"   && echo "  ✓ variable ASSETS_BUCKET = garassino-ml-artifacts"
gh variable set ASSETS_PREFIX -R "${REPO}" -b "deepmaze/"                && echo "  ✓ variable ASSETS_PREFIX = deepmaze/"
gh secret   set GCP_PROJECT_ID    -R "${REPO}" -b "garassino-ml"         && echo "  ✓ secret   GCP_PROJECT_ID = garassino-ml"
gh secret   set CLOUD_RUN_SERVICE -R "${REPO}" -b "deepmaze-backend"     && echo "  ✓ secret   CLOUD_RUN_SERVICE = deepmaze-backend"
# garassino-op's WIF provider — discovered via `gcloud iam workload-identity-pools providers list`
gh secret   set WIF_PROVIDER -R "${REPO}" -b "projects/634336216563/locations/global/workloadIdentityPools/gh-actions/providers/github" \
    && echo "  ✓ secret   WIF_PROVIDER = projects/634336216563/.../providers/github"

# Drop the deprecated GAR_REPO (no-op if it doesn't exist).
if gh secret list -R "${REPO}" 2>/dev/null | awk '{print $1}' | grep -q '^GAR_REPO$'; then
    gh secret delete GAR_REPO -R "${REPO}"
    echo "  ✓ secret   GAR_REPO removed (deprecated under GHCR-based deploy)"
fi

echo

# ---------------------------------------------------------------------------
# User-supplied — env var first, prompt second, skip on empty.
# ---------------------------------------------------------------------------

prompt_set() {
    local kind="$1"     # secret | variable
    local name="$2"
    local explain="$3"
    local envval="${!name:-}"

    if [ -n "${envval}" ]; then
        gh "${kind}" set "${name}" -R "${REPO}" -b "${envval}"
        echo "  ✓ ${kind}   ${name} (from env)"
        return
    fi

    echo
    echo "  ${name} — ${explain}"
    if [ "${kind}" = "secret" ]; then
        read -r -s -p "    value (blank to skip): " val; echo
    else
        read -r -p "    value (blank to skip): " val
    fi
    if [ -n "${val}" ]; then
        gh "${kind}" set "${name}" -R "${REPO}" -b "${val}"
        echo "  ✓ ${kind}   ${name}"
    else
        echo "    skipped"
    fi
}

echo "User-supplied values (press Enter to skip any):"
prompt_set variable CORS_ORIGINS         "comma-separated frontend origins, or '*' for demo"
prompt_set secret   WIF_SERVICE_ACCOUNT  "runtime SA email (Terraform output 'sa_email' — typically deepmaze-backend@garassino-ml.iam.gserviceaccount.com)"
prompt_set secret   CLOUD_RUN_SA_EMAIL   "runtime SA email — same as WIF_SERVICE_ACCOUNT"
prompt_set secret   TELEGRAM_BOT_TOKEN   "@BotFather token (optional)"
prompt_set secret   TELEGRAM_CHAT_ID     "your Telegram chat id (optional)"
prompt_set secret   SLACK_WEBHOOK_URL    "Slack webhook (optional)"
prompt_set secret   ANTHROPIC_API_KEY    "Anthropic API key for RunPod self-improve (optional)"
prompt_set secret   RUNPOD_API_KEY       "RunPod API key for GHA-driven pod creation (optional)"

echo
echo "Current state on ${REPO}:"
echo "---"
gh secret   list -R "${REPO}"
echo "---"
gh variable list -R "${REPO}"
