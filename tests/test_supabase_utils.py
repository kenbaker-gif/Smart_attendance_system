import importlib
from pathlib import Path

import pytest

from app.utils import supabase_utils as sutils


class FakeStorage:
    def __init__(self, files):
        self._files = files

    def from_(self, bucket_name):
        return self

    def list(self, prefix, options=None):
        return self._files

    def download(self, path):
        return b"fake-bytes"


class FakeClient:
    def __init__(self, files):
        self.storage = FakeStorage(files)


def test_download_all_supabase_images(tmp_path, monkeypatch):
    files = [{"name": "2400102415/1.jpg"}]
    fake_client = FakeClient(files)

    # Patch init_supabase_client to return our fake client
    monkeypatch.setattr(sutils, "init_supabase_client", lambda url, key: fake_client)

    ok = sutils.download_all_supabase_images(
        supabase_url="https://example",
        supabase_key="key",
        bucket_name="bucket",
        local_root=str(tmp_path),
        clear_local=True,
    )

    assert ok is True
    out_file = tmp_path / "2400102415" / "1.jpg"
    assert out_file.exists()
    assert out_file.read_bytes() == b"fake-bytes"
