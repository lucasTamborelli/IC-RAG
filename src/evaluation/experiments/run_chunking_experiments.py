"""
Ablação de chunking e top-k com voyage-multilingual + hybrid RRF.

Para cada configuração de chunk size, overlap e top-k, popula índices
FAISS/BM25 e avalia contra o benchmark sintético. Gera tabela LaTeX
com média ± erro-padrão da média (SEM).

Uso:
    python -m src.evaluation.experiments.run_chunking_experiments
    python -m src.evaluation.experiments.run_chunking_experiments --skip-populate
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.db_populator import populate_databases
from src.retrieval.database_loader import load_databases
from src.evaluation.experiments.run_experiments import (
    BENCHMARK_PATH,
    RESULTS_DIR,
    load_benchmark,
    run_retrieval_experiment,
)
from src.config import (
    ANALYSIS_DIR,
    BEST_EMBEDDING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    METRICS,
    VECTOR_DB_DIR,
)

EMBEDDING_KEY = BEST_EMBEDDING
OUTPUT_TEX = f"{ANALYSIS_DIR}/chunking_ablation_voyage.tex"

BASELINE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
BASELINE_OVERLAP = DEFAULT_OVERLAP
BASELINE_TOP_K = DEFAULT_TOP_K

# Eixo 1: chunk_size com overlap proporcional (~12–15%, k=3)
# Eixo 2: overlap com cs=1000 fixo (k=3)
# Eixo 3: top-k com baseline cs=1000, ov=150
CHUNKING_CONFIGS = [
    {"chunk_size": 256,  "overlap": 32,  "top_k": 3},
    {"chunk_size": 512,  "overlap": 64,  "top_k": 3},
    {"chunk_size": 1000, "overlap": 150, "top_k": 3},
    {"chunk_size": 2048, "overlap": 256, "top_k": 3},
    {"chunk_size": 1000, "overlap": 0,   "top_k": 3},
    {"chunk_size": 1000, "overlap": 64,  "top_k": 3},
    {"chunk_size": 1000, "overlap": 256, "top_k": 3},
    {"chunk_size": 1000, "overlap": 150, "top_k": 2},
    {"chunk_size": 1000, "overlap": 150, "top_k": 5},
    {"chunk_size": 1000, "overlap": 150, "top_k": 10},
]


def config_label(cfg: dict) -> str:
    return f"cs{cfg['chunk_size']}_ov{cfg['overlap']}_k{cfg['top_k']}"


def index_label(cfg: dict) -> str:
    return f"cs{cfg['chunk_size']}_ov{cfg['overlap']}"


def db_dir_for(cfg: dict) -> str:
    return f"{VECTOR_DB_DIR}/exp_chunk_voyage_{index_label(cfg)}"


def sem(std: float, n: int) -> float:
    return std / math.sqrt(n)


def fmt_pt(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def ensure_index(cfg: dict, skip_populate: bool) -> str:
    candidates = [
        db_dir_for(cfg),
        f"{VECTOR_DB_DIR}/exp_chunk_voyage_{config_label(cfg)}",
    ]
    for db_dir in candidates:
        faiss_path = os.path.join(db_dir, "faiss_index", "index.faiss")
        if os.path.exists(faiss_path):
            print(f"  Índice existente: {db_dir}")
            return db_dir

    db_dir = db_dir_for(cfg)
    if skip_populate:
        raise FileNotFoundError(
            f"Índice não encontrado: {db_dir}. Rode sem --skip-populate."
        )

    print(
        f"  Populando {db_dir} "
        f"(cs={cfg['chunk_size']}, ov={cfg['overlap']})..."
    )
    populate_databases(
        embedding_key=EMBEDDING_KEY,
        output_directory=db_dir,
        chunk_size=cfg["chunk_size"],
        overlap=cfg["overlap"],
    )
    return db_dir


def run_chunking_experiment(
    benchmark: list[dict],
    skip_populate: bool,
) -> list[dict]:
    results = []

    for cfg in CHUNKING_CONFIGS:
        label = config_label(cfg)
        print(f"\nConfig: {label}")
        db_dir = ensure_index(cfg, skip_populate)

        faiss_db, bm25_retriever = load_databases(db_dir, EMBEDDING_KEY)
        result = run_retrieval_experiment(
            faiss_db,
            bm25_retriever,
            benchmark,
            strategy="hybrid_rrf",
            top_k=cfg["top_k"],
        )

        row = {
            "experiment": "chunking_ablation_voyage",
            "embedding": EMBEDDING_KEY,
            "strategy": "hybrid_rrf",
            "label": label,
            **cfg,
            **result["aggregated"],
            "by_type": result["by_type"],
            "per_query": result["per_query"],
        }
        results.append(row)
        print(
            f"    Hit Rate: {row['hit_rate']['mean']:.3f} | "
            f"MRR: {row['mrr']['mean']:.3f}"
        )

    return results


def results_to_latex_sem(results: list[dict], n: int = 28) -> str:
    metric_cols = {
        "hit_rate": "Hit Rate",
        "mrr": "MRR",
        "context_precision": "Ctx. Prec.",
        "context_recall": "Ctx. Recall",
    }

    best_per_metric = {
        m: max(r[m]["mean"] for r in results)
        for m in METRICS
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ablação de parâmetros de chunking e top-$k$ "
        r"(estratégia: hybrid RRF, \textit{Voyage multilíngue})}",
        r"\label{tab:chunking_ablation_voyage}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        "Chunk Size & Overlap & Top-$k$ & Hit Rate & MRR & Ctx. Prec. & Ctx. Recall \\\\",
        r"\midrule",
    ]

    for r in results:
        cells = [
            str(r["chunk_size"]),
            str(r["overlap"]),
            str(r["top_k"]),
        ]
        for m in METRICS:
            mean = r[m]["mean"]
            err = sem(r[m]["std"], n)
            cell = f"{fmt_pt(mean)} $\\pm$ {fmt_pt(err)}"
            if mean == best_per_metric[m]:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption*{\footnotesize Valores expressos como média $\pm$ erro-padrão da média "
        r"(SEM), calculado sobre as 28 perguntas do benchmark.}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def save_results(results: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"chunking_ablation_voyage_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados JSON: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Ablação de chunking com voyage-multilingual + tabela LaTeX (SEM)"
    )
    parser.add_argument("--benchmark", default=BENCHMARK_PATH)
    parser.add_argument(
        "--skip-populate",
        action="store_true",
        help="Usa índices existentes; falha se algum não existir",
    )
    parser.add_argument("--output-tex", default=OUTPUT_TEX)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    print(
        f"Benchmark: {len(benchmark)} perguntas | "
        f"Embedding: {EMBEDDING_KEY} | Estratégia: hybrid_rrf"
    )

    results = run_chunking_experiment(benchmark, skip_populate=args.skip_populate)
    save_results(results)

    latex = results_to_latex_sem(results)
    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Tabela LaTeX: {args.output_tex}\n")
    print(latex)


if __name__ == "__main__":
    main()
