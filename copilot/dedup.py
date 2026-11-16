"""Cross-source duplicate grouping: near-identical embeddings AND similar titles."""
import re

import numpy as np


def _tokens(title: str) -> set[str]:
    return set(re.findall(r"\w+", title.lower()))


def group_duplicates(
    ids: list[str], vectors: np.ndarray, titles: list[str], threshold: float = 0.95
) -> dict[str, str]:
    """Map each id to its group id (smallest id in the group)."""
    v = np.asarray(vectors, dtype="float32")
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    # ponytail: O(n²) over ~10k vacancies, use ANN if the corpus grows past ~50k
    sim = v @ v.T
    toks = [_tokens(t) for t in titles]
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(sim > threshold, k=1))):
        a, b = toks[i], toks[j]
        if a | b and len(a & b) / len(a | b) >= 0.5:
            parent[find(i)] = find(j)
    smallest: dict[int, str] = {}
    for i, id_ in enumerate(ids):
        r = find(i)
        smallest[r] = min(smallest.get(r, id_), id_)
    return {id_: smallest[find(i)] for i, id_ in enumerate(ids)}
