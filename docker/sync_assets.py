"""Background asset sync from GCS for the prod entrypoint.

Run as: `python docker/sync_assets.py` with ASSETS_BUCKET in env.
Walks gs://${ASSETS_BUCKET}/ and writes each blob to /app/assets/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    bucket_name = os.environ.get("ASSETS_BUCKET")
    if not bucket_name:
        return 0
    try:
        from google.cloud import storage
    except ImportError:
        print("[sync_assets] google-cloud-storage not installed; skip", file=sys.stderr)
        return 0

    dst_root = Path(os.environ.get("ASSETS_DIR", "/app/assets"))
    dst_root.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    n = 0
    for blob in bucket.list_blobs():
        if blob.name.endswith("/"):
            continue
        dst = dst_root / blob.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))
        n += 1
    print(f"[sync_assets] {n} files from gs://{bucket_name}/ → {dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
