from huggingface_hub import HfApi


def publish(index_dir: str, repo_id: str, tag: str | None = None, api: HfApi | None = None) -> str:
    """Upload the index folder to the private HF dataset; optionally tag the commit. Returns the sha."""
    api = api or HfApi()  # token from HF_TOKEN env
    # Refuse before uploading: snapshot tags are never moved or overwritten.
    if tag and tag in {t.name for t in api.list_repo_refs(repo_id, repo_type="dataset").tags}:
        raise ValueError(f"tag {tag} already exists on {repo_id}; snapshot tags are never moved")
    info = api.upload_folder(folder_path=index_dir, repo_id=repo_id, repo_type="dataset",
                            delete_patterns="*")  # mirror the folder; tags keep history
    if tag:
        api.create_tag(repo_id, tag=tag, revision=info.oid, repo_type="dataset")
    return info.oid
