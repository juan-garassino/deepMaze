#!/usr/bin/env sh
# Replace ${API_BASE_URL} inside config.js at container start.
set -e
CONFIG=/usr/share/nginx/html/config.js
if [ -f "$CONFIG" ]; then
  # use envsubst, but limit to the API_BASE_URL var so other $ in the file
  # (none currently, but safer) are not eaten.
  TMP=$(mktemp)
  envsubst '${API_BASE_URL}' < "$CONFIG" > "$TMP"
  mv "$TMP" "$CONFIG"
fi
