"""bge-m3 embeddings and bge-reranker-v2-m3 reranking. Models load lazily on first call."""
from functools import cache

import numpy as np

# ponytail: 8192 is bge-m3's max; revisit with real `stats` p90 (CPU embed time grows with length)
MAX_SEQ_LENGTH = 8192
# Reranker sees up to 30 (query, text) pairs per search on a free 2-vCPU CPU: keep it short.
RERANK_MAX_LENGTH = 512
EMBED_MODEL = "BAAI/bge-m3"
# Stored with the index; vectors are reused only when it matches. Bump "tpl" when copilot.index.text_of changes.
EMBED_FINGERPRINT = f"{EMBED_MODEL}|max_seq={MAX_SEQ_LENGTH}|tpl=1"


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(EMBED_MODEL)
    m.max_seq_length = MAX_SEQ_LENGTH
    return m


@cache
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=RERANK_MAX_LENGTH)


def embed(texts: list[str]) -> np.ndarray:
    return _embedder().encode(texts, normalize_embeddings=True, batch_size=8).astype("float32")


def rerank(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _reranker().predict([(query, t) for t in texts])]
