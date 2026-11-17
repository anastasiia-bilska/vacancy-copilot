from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from copilot.publish import publish


def fake_api(existing_tags=()):
    api = MagicMock()
    api.upload_folder.return_value = SimpleNamespace(oid="abc123")
    api.list_repo_refs.return_value = SimpleNamespace(tags=[SimpleNamespace(name=t) for t in existing_tags])
    return api


def test_publish_uploads_dataset_without_tag():
    api = fake_api()
    assert publish("data/index", "me/idx", api=api) == "abc123"
    kw = api.upload_folder.call_args.kwargs
    assert kw["repo_type"] == "dataset" and kw["repo_id"] == "me/idx" and kw["folder_path"] == "data/index"
    assert kw["delete_patterns"] == "*"
    api.create_tag.assert_not_called()


def test_publish_tags_commit():
    api = fake_api()
    publish("data/index", "me/idx", tag="eval-2026-11-16", api=api)
    kw = api.create_tag.call_args.kwargs
    assert kw["tag"] == "eval-2026-11-16" and kw["revision"] == "abc123" and kw["repo_type"] == "dataset"


def test_existing_tag_raises_before_upload():
    api = fake_api(existing_tags=["eval-2026-11-16"])
    with pytest.raises(ValueError, match="already exists"):
        publish("data/index", "me/idx", tag="eval-2026-11-16", api=api)
    api.upload_folder.assert_not_called()
    api.create_tag.assert_not_called()
