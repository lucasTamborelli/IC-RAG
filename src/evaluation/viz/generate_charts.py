"""
Gera graficos a partir dos resultados experimentais.

Saida: PNGs em data/eval/analysis/figures/

Uso:
    python -m src.evaluation.viz.generate_charts
"""

import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.config import FIG_DIR, RAG_JSON as JUDGE_PATH, RESULTS_DIR

COLORS = {
    "openai-small": "#74AA9C",
    "openai-large": "#10A37F",
    "voyage-multilingual": "#6C5CE7",
}

STRATEGY_LABELS = {
    "semantic": "Semantic",
    "keyword": "Keyword\n(BM25)",
    "hybrid_weighted": "Hybrid\nWeighted",
    "hybrid_rrf": "Hybrid\nRRF",
}

METRIC_LABELS = {
    "hit_rate": "Hit Rate @k",
    "mrr": "MRR",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def load_latest(experiment_name):
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR)
         if f.startswith(experiment_name) and f.endswith(".json") and os.path.getsize(os.path.join(RESULTS_DIR, f)) > 10],
        reverse=True,
    )
    if not files:
        return []
    with open(os.path.join(RESULTS_DIR, files[0]), "r", encoding="utf-8") as f:
        return json.load(f)


def fig_embedding_grouped_bar():
    """Bar chart agrupado: embedding x estrategia, metrica = Hit Rate."""
    results = load_latest("embedding_comparison")
    if not results:
        return

    embeddings = list(dict.fromkeys(r["embedding"] for r in results))
    strategies = list(dict.fromkeys(r["strategy"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, metric in enumerate(["hit_rate", "mrr"]):
        ax = axes[ax_idx]
        x = np.arange(len(strategies))
        width = 0.25
        offsets = np.linspace(-width, width, len(embeddings))

        for i, emb in enumerate(embeddings):
            means = []
            stds = []
            for strat in strategies:
                match = [r for r in results if r["embedding"] == emb and r["strategy"] == strat]
                if match:
                    means.append(match[0][metric]["mean"])
                    stds.append(match[0][metric]["std"])
                else:
                    means.append(0)
                    stds.append(0)

            bars = ax.bar(
                x + offsets[i], means, width * 0.9,
                yerr=stds, capsize=3,
                label=emb, color=COLORS.get(emb, "#999"),
                edgecolor="white", linewidth=0.5,
                error_kw={"linewidth": 1, "alpha": 0.6},
            )
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Comparacao de Modelos de Embedding por Estrategia de Retrieval", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIG_DIR, "embedding_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_embedding_heatmap():
    """Heatmap: embedding x estrategia, valor = Hit Rate."""
    results = load_latest("embedding_comparison")
    if not results:
        return

    embeddings = list(dict.fromkeys(r["embedding"] for r in results))
    strategies = list(dict.fromkeys(r["strategy"] for r in results))

    matrix = np.zeros((len(embeddings), len(strategies)))
    for r in results:
        i = embeddings.index(r["embedding"])
        j = strategies.index(r["strategy"])
        matrix[i, j] = r["hit_rate"]["mean"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], fontsize=10)
    ax.set_yticks(range(len(embeddings)))
    ax.set_yticklabels(embeddings, fontsize=10)

    for i in range(len(embeddings)):
        for j in range(len(strategies)):
            val = matrix[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Hit Rate @k", fontsize=10)
    ax.set_title("Hit Rate por Embedding e Estrategia", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "embedding_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_fusion_comparison():
    """Bar chart horizontal: estrategias de fusao."""
    results = load_latest("fusion_comparison")
    if not results:
        return

    labels = [r["strategy"].replace("hybrid_weighted_", "w=").replace("hybrid_rrf", "RRF") for r in results]
    hr = [r["hit_rate"]["mean"] for r in results]
    mrr = [r["mrr"]["mean"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(labels))
    height = 0.35

    bars1 = ax.barh(y - height / 2, hr, height, label="Hit Rate", color="#10A37F", edgecolor="white")
    bars2 = ax.barh(y + height / 2, mrr, height, label="MRR", color="#6C5CE7", edgecolor="white")

    for bar, val in zip(bars1, hr):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, mrr):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Comparacao de Estrategias de Fusao Hybrid", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fusion_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_topk_tradeoff():
    """Line chart: top-k vs Hit Rate e Context Precision (trade-off)."""
    results = load_latest("chunking_ablation")
    if not results:
        return

    topk_results = [r for r in results if r["chunk_size"] == 1000 and r["overlap"] == 150]
    topk_results.sort(key=lambda r: r["top_k"])

    if len(topk_results) < 2:
        return

    ks = [r["top_k"] for r in topk_results]
    hr = [r["hit_rate"]["mean"] for r in topk_results]
    cp = [r["context_precision"]["mean"] for r in topk_results]
    cr = [r["context_recall"]["mean"] for r in topk_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_hr = "#10A37F"
    color_cp = "#E74C3C"
    color_cr = "#3498DB"

    ax1.plot(ks, hr, "o-", color=color_hr, linewidth=2.5, markersize=8, label="Hit Rate", zorder=3)
    ax1.plot(ks, cr, "s--", color=color_cr, linewidth=2, markersize=7, label="Context Recall", zorder=3)
    ax1.plot(ks, cp, "^--", color=color_cp, linewidth=2, markersize=7, label="Context Precision", zorder=3)

    for k, h, c, p in zip(ks, hr, cr, cp):
        ax1.annotate(f"{h:.2f}", (k, h), textcoords="offset points", xytext=(0, 10),
                     fontsize=8.5, fontweight="bold", color=color_hr, ha="center")
        ax1.annotate(f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, -14),
                     fontsize=8.5, fontweight="bold", color=color_cp, ha="center")

    ax1.set_xlabel("Top-k (documentos recuperados)", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(ks)
    ax1.legend(fontsize=10, loc="center right")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Trade-off: Top-k vs Hit Rate e Precision", fontsize=13, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "topk_tradeoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_chunksize_comparison():
    """Bar chart: chunk size vs metricas (fixando top_k=3, variando chunk_size)."""
    results = load_latest("chunking_ablation")
    if not results:
        return

    cs_results = [r for r in results if r["top_k"] == 3 and r["overlap"] != 0 and r["overlap"] != 300]
    cs_results.sort(key=lambda r: r["chunk_size"])

    if len(cs_results) < 2:
        return

    labels = [str(r["chunk_size"]) for r in cs_results]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    metrics_to_plot = ["hit_rate", "mrr", "context_precision", "context_recall"]
    colors_m = ["#10A37F", "#6C5CE7", "#E74C3C", "#3498DB"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_m)):
        vals = [r[metric]["mean"] for r in cs_results]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.9,
                      label=METRIC_LABELS.get(metric, metric), color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Chunk Size (caracteres)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Impacto do Chunk Size nas Metricas de Retrieval (Hybrid RRF, top-k=3)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "chunksize_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_llm_judge_by_type():
    """Bar chart agrupado: tipo de pergunta vs dimensoes do LLM-as-Judge."""
    if not os.path.exists(JUDGE_PATH):
        return

    with open(JUDGE_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    valid = [r for r in results if r.get("faithfulness") is not None]
    if not valid:
        return

    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("tipo", "?")].append(r)

    tipos = sorted(by_type.keys())
    dims = ["faithfulness", "answer_relevancy", "correctness"]
    dim_labels = ["Faithfulness", "Answer\nRelevancy", "Correctness"]
    colors_d = ["#10A37F", "#6C5CE7", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(tipos))
    width = 0.25

    for i, (dim, label, color) in enumerate(zip(dims, dim_labels, colors_d)):
        means = []
        for tipo in tipos:
            vals = [r[dim] for r in by_type[tipo] if r.get(dim) is not None]
            means.append(sum(vals) / len(vals) if vals else 0)

        bars = ax.bar(x + (i - 1) * width, means, width * 0.9,
                      label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    overall_means = []
    for dim in dims:
        vals = [r[dim] for r in valid if r.get(dim) is not None]
        overall_means.append(sum(vals) / len(vals) if vals else 0)

    ax.axhline(y=overall_means[0], color=colors_d[0], linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tipos], fontsize=11)
    ax.set_xlabel("Tipo de Pergunta", fontsize=11)
    ax.set_ylabel("Score (1-5)", fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Avaliacao LLM-as-Judge por Tipo de Pergunta", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "llm_judge_by_type.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def fig_radar_best_configs():
    """Radar chart comparando as 3 melhores configuracoes."""
    results = load_latest("embedding_comparison")
    if not results:
        return

    sorted_r = sorted(results, key=lambda r: r["hit_rate"]["mean"], reverse=True)
    top3 = sorted_r[:3]

    metrics = ["hit_rate", "mrr", "context_precision", "context_recall"]
    labels = ["Hit Rate", "MRR", "Ctx Precision", "Ctx Recall"]
    num_vars = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors_top = ["#10A37F", "#6C5CE7", "#E74C3C"]
    for i, r in enumerate(top3):
        values = [r[m]["mean"] for m in metrics]
        values += values[:1]
        config_label = f"{r['embedding']} + {r['strategy']}"

        ax.plot(angles, values, "o-", linewidth=2, markersize=6,
                label=config_label, color=colors_top[i])
        ax.fill(angles, values, alpha=0.1, color=colors_top[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Top-3 Configuracoes de Retrieval", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "radar_top3.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Gerando graficos...\n")

    fig_embedding_grouped_bar()
    fig_embedding_heatmap()
    fig_fusion_comparison()
    fig_topk_tradeoff()
    fig_chunksize_comparison()
    fig_llm_judge_by_type()
    fig_radar_best_configs()

    print(f"\nTodos os graficos salvos em: {FIG_DIR}")


if __name__ == "__main__":
    main()
