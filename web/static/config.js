// API base URL.
// In Docker split mode: nginx envsubst replaces ${API_BASE_URL} at startup.
// In local/monolithic mode: the literal string remains, and we fall back to
// same-origin (empty prefix).
(function () {
  const raw = "${API_BASE_URL}";
  window.API_BASE_URL = raw.startsWith("$") ? "" : raw;
})();
