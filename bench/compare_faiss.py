"""Run FAISS's HNSW (IndexHNSWFlat) on the same dataset and parameters as
quiver-bench, and write results in the same JSON shape for plot.py.

    python bench/compare_faiss.py --data path/to/sift --out bench/results/faiss-sift1m.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import faiss
import numpy as np


def read_vecs(path: str, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    rows = raw.reshape(-1, dim + 1)
    return rows[:, 1:].copy().view(dtype)


def recall_at_k(truth: np.ndarray, found: np.ndarray, k: int) -> float:
    hits = sum(len(set(t[:k]) & set(f[:k])) for t, f in zip(truth, found))
    return hits / (len(truth) * k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory with sift_base/query/groundtruth files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", default="10,16,24,32,48,64,96,128,192,256,384,512")
    args = parser.parse_args()

    base = read_vecs(os.path.join(args.data, "sift_base.fvecs"), np.float32)
    query = read_vecs(os.path.join(args.data, "sift_query.fvecs"), np.float32)
    truth = read_vecs(os.path.join(args.data, "sift_groundtruth.ivecs"), np.int32)[:, : args.k]
    n, dim = base.shape
    threads = os.cpu_count() or 1
    print(f"faiss {faiss.__version__} | n={n} dim={dim} queries={len(query)} threads={threads}")

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(dim, args.M)
    index.hnsw.efConstruction = args.efc
    start = time.perf_counter()
    index.add(base)
    build_seconds = time.perf_counter() - start
    print(f"build: {build_seconds:.1f} s ({n / build_seconds:,.0f} vectors/s)")

    runs = []
    print(f"{'ef':>6} {'recall':>9} {'QPS(1 thr)':>12} {'QPS(all)':>12}")
    for ef in [int(x) for x in args.ef.split(",")]:
        index.hnsw.efSearch = ef
        # Single thread, one query per call: comparable to quiver-bench's latency loop.
        faiss.omp_set_num_threads(1)
        found = np.empty((len(query), args.k), dtype=np.int64)
        start = time.perf_counter()
        for i in range(len(query)):
            _, found[i] = index.search(query[i : i + 1], args.k)
        qps_single = len(query) / (time.perf_counter() - start)

        # All cores, one batched call: FAISS's best case for throughput.
        faiss.omp_set_num_threads(threads)
        rounds = 3
        start = time.perf_counter()
        for _ in range(rounds):
            index.search(query, args.k)
        qps_all = rounds * len(query) / (time.perf_counter() - start)

        recall = recall_at_k(truth, found, args.k)
        print(f"{ef:>6} {recall:>9.4f} {qps_single:>12,.0f} {qps_all:>12,.0f}")
        runs.append({"ef": ef, "recall": recall, "qps_single_thread": qps_single, "qps_all_threads": qps_all})

    report = {
        "engine": "faiss",
        "version": faiss.__version__,
        "dataset": "sift1m",
        "n": n,
        "dim": dim,
        "queries": len(query),
        "k": args.k,
        "M": args.M,
        "ef_construction": args.efc,
        "threads": threads,
        "build_seconds": build_seconds,
        "runs": runs,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
