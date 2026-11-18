"""Semantic search over a small corpus, with metadata filtering.

Embeds documents with character n-gram LSA (TF-IDF + truncated SVD from
scikit-learn), which copes with words it has never seen, so it runs
offline in seconds. Swap `embed` for any sentence-embedding model (e.g.
sentence-transformers) and nothing else changes.

    quiver-server --port 7070 &
    pip install scikit-learn ./clients/python
    python examples/semantic_search.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "python"))

from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from quiver_client import Quiver  # noqa: E402

CORPUS = [
    ("db", "B-trees keep keys sorted on disk so range scans touch few pages."),
    ("db", "A write-ahead log lets a database recover committed transactions after a crash."),
    ("db", "Log-structured merge trees batch writes in memory and flush sorted runs to disk."),
    ("db", "MVCC gives each transaction a consistent snapshot without blocking writers."),
    ("db", "Query planners estimate the cost of index scans versus full table scans."),
    ("db", "Checkpointing bounds recovery time by persisting a snapshot and truncating the log."),
    ("ml", "Transformers use self-attention to weigh every token against every other token."),
    ("ml", "Sentence embeddings map text to dense vectors where similar meanings sit close together."),
    ("ml", "Retrieval-augmented generation fetches relevant passages before the model answers."),
    ("ml", "Gradient descent updates weights in the direction that reduces the loss."),
    ("ml", "Contrastive learning pulls matching pairs together and pushes others apart."),
    ("ml", "Quantization stores model weights in fewer bits to save memory and bandwidth."),
    ("systems", "SIMD instructions apply one operation to many floats in a single cycle."),
    ("systems", "Cache misses dominate when a program chases pointers through random memory."),
    ("systems", "A mutex serialises access to shared state between threads."),
    ("systems", "Lock striping spreads contention over many locks instead of one global lock."),
    ("systems", "fsync forces buffered file data to stable storage before returning."),
    ("systems", "Prefetching loads memory into cache before the CPU asks for it."),
    ("space", "A CubeSat downlinks images in frames protected by CRC checksums."),
    ("space", "Ground stations request retransmission of frames lost during a satellite pass."),
    ("space", "NDVI from red and near-infrared bands measures vegetation health from orbit."),
    ("space", "Low Earth orbit satellites complete a revolution in about ninety minutes."),
]

QUERIES = [
    ("how does a database survive a power failure?", None),
    ("finding similar documents with vectors", None),
    ("making code faster on modern CPUs", {"topic": "systems"}),
    ("detecting corrupted data", {"topic": {"$in": ["space", "db"]}}),
]


def main() -> None:
    texts = [text for _, text in CORPUS]
    embedder = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        TruncatedSVD(n_components=20, random_state=0),
    )
    vectors = embedder.fit_transform(texts)

    def embed(text: str):
        return embedder.transform([text])[0]

    db = Quiver(os.environ.get("QUIVER_URL", "http://127.0.0.1:7070"))
    try:
        db.drop_collection("demo-docs")
    except Exception:
        pass
    docs = db.create_collection("demo-docs", dim=vectors.shape[1], metric="cosine", sync="flush")
    docs.upsert(
        {"id": f"doc-{i}", "vector": vectors[i], "metadata": {"topic": topic, "text": text}}
        for i, (topic, text) in enumerate(CORPUS)
    )
    print(f"indexed {docs.info()['points']} documents\n")

    for query, flt in QUERIES:
        result = docs.search(embed(query), k=3, filter=flt)
        suffix = f"   filter={flt}" if flt else ""
        print(f"Q: {query}{suffix}")
        print(f"   plan={result.plan['strategy']}  took={result.took_ms:.2f} ms")
        for hit in result:
            print(f"   {hit.score:+.3f}  [{hit.metadata['topic']:>7}] {hit.metadata['text']}")
        print()


if __name__ == "__main__":
    main()
