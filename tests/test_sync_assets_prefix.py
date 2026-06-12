"""Verify docker/sync_assets.py honors ASSETS_PREFIX.

Runs in-process with a fake google.cloud.storage so we don't need the real SDK
or network access. Exercises the two behaviors that matter for the post-2026-
06-07 architecture:
- with prefix → list_blobs gets prefix=...; destination paths have prefix stripped
- without prefix → list_blobs gets prefix=None; destination paths are unchanged
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def _load_sync_module(tmp_path: Path):
    """Load docker/sync_assets.py as a fresh module, with a fake google.cloud.storage in sys.modules."""
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "sync_assets_under_test",
        project_root / "docker" / "sync_assets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubBlob:
    def __init__(self, name: str, payload: bytes = b"x"):
        self.name = name
        self._payload = payload

    def download_to_filename(self, dst: str) -> None:
        Path(dst).write_bytes(self._payload)


class _StubBucket:
    def __init__(self, blobs: list[_StubBlob]):
        self._blobs = blobs
        self.list_kwargs: dict = {}

    def list_blobs(self, **kwargs):
        self.list_kwargs = kwargs
        prefix = kwargs.get("prefix")
        if not prefix:
            return list(self._blobs)
        return [b for b in self._blobs if b.name.startswith(prefix)]


class _StubClient:
    def __init__(self, bucket: _StubBucket):
        self._bucket = bucket

    def bucket(self, name: str) -> _StubBucket:
        return self._bucket


def _install_fake_storage(stub_bucket: _StubBucket):
    """Inject a stub google.cloud.storage so sync_assets.py's import succeeds without the SDK."""
    google_pkg = types.ModuleType("google")
    google_cloud_pkg = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = lambda: _StubClient(stub_bucket)
    google_cloud_pkg.storage = storage_mod
    google_pkg.cloud = google_cloud_pkg
    sys.modules["google"] = google_pkg
    sys.modules["google.cloud"] = google_cloud_pkg
    sys.modules["google.cloud.storage"] = storage_mod
    return storage_mod


def test_prefix_strips_destination_paths(tmp_path: Path, monkeypatch):
    blobs = [
        _StubBlob("deepmaze/model_a/config.json"),
        _StubBlob("deepmaze/model_a/model.pt"),
        _StubBlob("deepmaze/model_b/viz/replay.webp"),
        _StubBlob("other_project/model_x/config.json"),  # should NOT land
    ]
    bucket = _StubBucket(blobs)
    _install_fake_storage(bucket)

    dst = tmp_path / "assets"
    monkeypatch.setenv("ASSETS_BUCKET", "garassino-ml-artifacts")
    monkeypatch.setenv("ASSETS_PREFIX", "deepmaze/")
    monkeypatch.setenv("ASSETS_DIR", str(dst))

    mod = _load_sync_module(tmp_path)
    rc = mod.main()
    assert rc == 0
    assert bucket.list_kwargs == {"prefix": "deepmaze/"}
    # Paths land flat under ASSETS_DIR — prefix stripped.
    assert (dst / "model_a" / "config.json").exists()
    assert (dst / "model_a" / "model.pt").exists()
    assert (dst / "model_b" / "viz" / "replay.webp").exists()
    # Other project's blobs are not in the listing AND wouldn't land here.
    assert not (dst / "other_project").exists()


def test_no_prefix_preserves_full_paths(tmp_path: Path, monkeypatch):
    blobs = [
        _StubBlob("flat/model.pt"),
        _StubBlob("nested/dir/config.json"),
    ]
    bucket = _StubBucket(blobs)
    _install_fake_storage(bucket)

    dst = tmp_path / "assets"
    monkeypatch.setenv("ASSETS_BUCKET", "some-bucket")
    monkeypatch.delenv("ASSETS_PREFIX", raising=False)
    monkeypatch.setenv("ASSETS_DIR", str(dst))

    mod = _load_sync_module(tmp_path)
    rc = mod.main()
    assert rc == 0
    # No prefix → list_blobs called with prefix=None
    assert bucket.list_kwargs == {"prefix": None}
    # Full paths preserved
    assert (dst / "flat" / "model.pt").exists()
    assert (dst / "nested" / "dir" / "config.json").exists()


def test_no_bucket_env_returns_zero_noop(monkeypatch):
    """Without ASSETS_BUCKET the function exits 0 and does nothing."""
    monkeypatch.delenv("ASSETS_BUCKET", raising=False)
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "sync_assets_noop", project_root / "docker" / "sync_assets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0
