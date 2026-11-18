"""Draw the recall-vs-throughput chart used in the README from benchmark JSON.

    python bench/plot.py bench/results/quiver-sift1m.json bench/results/faiss-sift1m.json \
        --out docs/assets/sift1m-recall-qps.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1 and 2 of the validated reference palette (CVD-checked pair).
SERIES = {"quiver": "#2a78d6", "faiss": "#eb6834"}
LABELS = {"quiver": "Quiver (this project)", "faiss": "FAISS IndexHNSWFlat"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="qps_single_thread", choices=["qps_single_thread", "qps_all_threads"])
    args = parser.parse_args()

    reports = [json.load(open(path)) for path in args.reports]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for report in reports:
        engine = report["engine"]
        runs = sorted(report["runs"], key=lambda r: r["recall"])
        x = [r["recall"] for r in runs]
        y = [r[args.metric] for r in runs]
        color = SERIES.get(engine, "#6250d6")
        ax.plot(x, y, color=color, linewidth=2, label=LABELS.get(engine, engine), zorder=3)
        ax.scatter(x, y, s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        # Direct label at the high-recall end so identity is not colour-alone.
        ax.annotate(LABELS.get(engine, engine), (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")

    first = reports[0]
    threads = "1 thread" if args.metric == "qps_single_thread" else f"{first['threads']} threads"
    ax.set_title(f"SIFT1M: recall@{first['k']} vs throughput (M={first['M']}, efC={first['ef_construction']}, {threads})",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.set_xlabel(f"Recall@{first['k']}  (further right is more accurate)", color=MUTED, fontsize=9)
    ax.set_ylabel("Queries per second  (log scale, higher is faster)", color=MUTED, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    lo = min(r["recall"] for rep in reports for r in rep["runs"])
    ax.set_xlim(max(0.0, lo - 0.02), 1.06)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
