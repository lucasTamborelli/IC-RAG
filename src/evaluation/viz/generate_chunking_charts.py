"""
Gera gráficos de ablação de chunking a partir dos resultados voyage-multilingual.

Saída:
  - artigo/template_pibic_ita/figures/topk_tradeoff.png
  - artigo/template_pibic_ita/figures/chunksize_comparison.png
  - data/eval/analysis/figures/ (cópia)

Uso:
    python -m src.evaluation.viz.generate_chunking_charts
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    FIG_DIR as ANALYSIS_FIG_DIR,
    RESULTS_DIR,
)

ARTICLE_FIG_DIR = "./artigo/template_pibic_ita/figures"

BASELINE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
BASELINE_OVERLAP = DEFAULT_OVERLAP

CHUNK_SIZE_SWEEP = {(256, 32), (512, 64), (1000, 150), (2048, 256)}

METRIC_LABELS = {
    "hit_rate": "Hit Rate @k",
    "mrr": "MRR",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def load_latest_voyage_chunking() -> list[dict]:
    files = sorted(
        [
            f for f in os.listdir(RESULTS_DIR)
            if f.startswith("chunking_ablation_voyage") and f.endswith(".json")
            and os.path.getsize(os.path.join(RESULTS_DIR, f)) > 10
        ],
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            "Nenhum resultado chunking_ablation_voyage encontrado. "
            "Rode: python -m src.evaluation.experiments.run_chunking_experiments"
        )
    path = os.path.join(RESULTS_DIR, files[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig, filename: str) -> None:
    for out_dir in (ARTICLE_FIG_DIR, ANALYSIS_FIG_DIR):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  -> {path}")


def fig_topk_tradeoff(results: list[dict]) -> None:
    topk_results = [
        r for r in results
        if r["chunk_size"] == BASELINE_CHUNK_SIZE and r["overlap"] == BASELINE_OVERLAP
    ]
    topk_results.sort(key=lambda r: r["top_k"])

    if len(topk_results) < 2:
        print("  [SKIP] topk_tradeoff: dados insuficientes")
        return

    ks = [r["top_k"] for r in topk_results]
    hr = [r["hit_rate"]["mean"] for r in topk_results]
    cp = [r["context_precision"]["mean"] for r in topk_results]
    cr = [r["context_recall"]["mean"] for r in topk_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    color_hr = "#10A37F"
    color_cp = "#E74C3C"
    color_cr = "#3498DB"

    ax.plot(ks, hr, "o-", color=color_hr, linewidth=2.5, markersize=8, label="Hit Rate", zorder=3)
    ax.plot(ks, cr, "s--", color=color_cr, linewidth=2, markersize=7, label="Context Recall", zorder=3)
    ax.plot(ks, cp, "^--", color=color_cp, linewidth=2, markersize=7, label="Context Precision", zorder=3)

    for k, h, c, p in zip(ks, hr, cr, cp):
        ax.annotate(
            f"{h:.2f}", (k, h), textcoords="offset points", xytext=(0, 10),
            fontsize=8.5, fontweight="bold", color=color_hr, ha="center",
        )
        ax.annotate(
            f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, -14),
            fontsize=8.5, fontweight="bold", color=color_cp, ha="center",
        )

    ax.set_xlabel("Top-k (documentos recuperados)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(ks)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Trade-off: Top-k vs Hit Rate e Precision\n"
        f"(Voyage multilingue, hybrid RRF, chunk={BASELINE_CHUNK_SIZE}, overlap={BASELINE_OVERLAP})",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    _save_fig(fig, "topk_tradeoff.png")
    plt.close(fig)


def fig_chunksize_comparison(results: list[dict]) -> None:
    cs_results = [
        r for r in results
        if r["top_k"] == 3
        and (r["chunk_size"], r["overlap"]) in CHUNK_SIZE_SWEEP
    ]
    cs_results.sort(key=lambda r: r["chunk_size"])

    if len(cs_results) < 2:
        print("  [SKIP] chunksize_comparison: dados insuficientes")
        return

    labels = [str(r["chunk_size"]) for r in cs_results]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    metrics_to_plot = ["hit_rate", "mrr", "context_precision", "context_recall"]
    colors_m = ["#10A37F", "#6C5CE7", "#E74C3C", "#3498DB"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_m)):
        vals = [r[metric]["mean"] for r in cs_results]
        bars = ax.bar(
            x + (i - 1.5) * width,
            vals,
            width * 0.9,
            label=METRIC_LABELS.get(metric, metric),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Chunk Size (caracteres)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Impacto do Chunk Size nas Metricas de Retrieval\n"
        "(Voyage multilingue, hybrid RRF, top-k=3)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    _save_fig(fig, "chunksize_comparison.png")
    plt.close(fig)


def fig_overlap_comparison(results: list[dict]) -> None:
    """Gráfico extra: impacto do overlap (cs=1024, k=3)."""
    ov_results = [
        r for r in results
        if r["chunk_size"] == BASELINE_CHUNK_SIZE and r["top_k"] == 3
    ]
    ov_results.sort(key=lambda r: r["overlap"])

    if len(ov_results) < 2:
        print("  [SKIP] overlap_comparison: dados insuficientes")
        return

    labels = [str(r["overlap"]) for r in ov_results]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["hit_rate", "mrr", "context_precision", "context_recall"]
    colors_m = ["#10A37F", "#6C5CE7", "#E74C3C", "#3498DB"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_m)):
        vals = [r[metric]["mean"] for r in ov_results]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width * 0.9,
            label=METRIC_LABELS.get(metric, metric),
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Overlap (caracteres)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Impacto do Overlap nas Metricas de Retrieval\n"
        f"(Voyage multilingue, hybrid RRF, chunk={BASELINE_CHUNK_SIZE}, top-k=3)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    _save_fig(fig, "overlap_comparison.png")
    plt.close(fig)


def main():
    results = load_latest_voyage_chunking()
    print(f"Carregado: {len(results)} configs (voyage-multilingual)\n")

    fig_topk_tradeoff(results)
    fig_chunksize_comparison(results)
    fig_overlap_comparison(results)

    print("\nGraficos de chunking atualizados.")


if __name__ == "__main__":
    main()
