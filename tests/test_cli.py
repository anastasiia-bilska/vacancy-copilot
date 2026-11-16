import pytest

import copilot.__main__ as cli


def test_eval_snapshot_downloads_tagged_index(monkeypatch):
    calls = []
    monkeypatch.setenv("HF_INDEX_REPO", "u/idx")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: calls.append((a, kw)))
    assert cli.download_snapshot("eval-2026-11-16") == "data/eval-eval-2026-11-16"
    assert calls == [(("u/idx",), {"repo_type": "dataset", "revision": "eval-2026-11-16",
                                   "local_dir": "data/eval-eval-2026-11-16"})]


def test_eval_snapshot_requires_repo(monkeypatch):
    monkeypatch.delenv("HF_INDEX_REPO", raising=False)
    with pytest.raises(SystemExit, match="HF_INDEX_REPO"):
        cli.download_snapshot("eval-x")
