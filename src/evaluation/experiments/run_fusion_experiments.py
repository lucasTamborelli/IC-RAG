"""
Comparação de estratégias de fusão hybrid com voyage-multilingual.

Roda o sweep de pesos sparse (weighted) + RRF contra o benchmark sintético
e gera tabela LaTeX com média ± erro-padrão da média (SEM).

Uso:
    python -m src.evaluation.experiments.run_fusion_experiments
    python -m src.evaluation.experiments.run_fusion_experiments --skip-populate

No Windows, se aparecer erro OMP/libiomp5, o script já define
KMP_DUPLICATE_LIB_OK=TRUE automaticamente.
"""

from __future__ import annotations

import os

# Evita crash por conflito libomp/libiomp5 no Windows (FAISS + PyTorch etc.)
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
from src.config import ANALYSIS_DIR, BEST_EMBEDDING, METRICS, VECTOR_DB_DIR

EMBEDDING_KEY = BEST_EMBEDDING
DB_DIR = f"{VECTOR_DB_DIR}/exp_voyage-multilingual"
OUTPUT_TEX = f"{ANALYSIS_DIR}/fusion_comparison_voyage.tex"

FUSION_ROWS = [
    ("weighted $w=0.0$ (só denso)", 0.0, "hybrid_weighted"),
    ("weighted $w=0.3$", 0.3, "hybrid_weighted"),
    ("weighted $w=0.5$", 0.5, "hybrid_weighted"),
    ("weighted $w=0.7$", 0.7, "hybrid_weighted"),
    ("weighted $w=1.0$ (só esparso)", 1.0, "hybrid_weighted"),
    ("RRF ($k=60$)", None, "hybrid_rrf"),
]


def sem(std: float, n: int) -> float:
    return std / math.sqrt(n)


def fmt_pt(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def run_fusion_experiment(benchmark: list[dict]) -> list[dict]:
    faiss_db, bm25_retriever = load_databases(DB_DIR, EMBEDDING_KEY)
    results = []

    for label, sparse_weight, strategy in FUSION_ROWS:
        print(f"  {label}...")
        if strategy == "hybrid_weighted":
            result = run_retrieval_experiment(
                faiss_db,
                bm25_retriever,
                benchmark,
                strategy="hybrid_weighted",
                sparse_weight=sparse_weight,
            )
            row = {
                "experiment": "fusion_comparison_voyage",
                "embedding": EMBEDDING_KEY,
                "strategy": f"hybrid_weighted_sw{sparse_weight}",
                "sparse_weight": sparse_weight,
                "label": label,
            }
        else:
            result = run_retrieval_experiment(
                faiss_db,
                bm25_retriever,
                benchmark,
                strategy="hybrid_rrf",
            )
            row = {
                "experiment": "fusion_comparison_voyage",
                "embedding": EMBEDDING_KEY,
                "strategy": "hybrid_rrf",
                "sparse_weight": None,
                "label": label,
            }

        row.update(result["aggregated"])
        row["by_type"] = result["by_type"]
        row["per_query"] = result["per_query"]
        results.append(row)
        print(
            f"    Hit Rate: {row['hit_rate']['mean']:.3f} | "
            f"MRR: {row['mrr']['mean']:.3f}"
        )

    return results


def results_to_latex_sem(results: list[dict], n: int = 28) -> str:
    col_headers = {
        "hit_rate": "Hit Rate",
        "mrr": "MRR",
        "context_precision": "Ctx. Prec.",
        "context_recall": "Ctx. Recall",
    }

    best_per_col = {
        m: max(r[m]["mean"] for r in results)
        for m in METRICS
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Comparação de estratégias de fusão hybrid "
        r"(\textit{Voyage multilíngue}, chunk size = 1000, overlap = 150, top-$k$ = 3)}",
        r"\label{tab:fusion_comparison_voyage}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Configuração & "
        + " & ".join(col_headers[m] for m in METRICS)
        + r" \\",
        r"\midrule",
    ]

    for r in results:
        cells = [r["label"]]
        for m in METRICS:
            mean = r[m]["mean"]
            err = sem(r[m]["std"], n)
            cell = f"{fmt_pt(mean)} $\\pm$ {fmt_pt(err)}"
            if mean == best_per_col[m]:
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
    path = os.path.join(RESULTS_DIR, f"fusion_comparison_voyage_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados JSON: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Fusão hybrid com voyage-multilingual + tabela LaTeX (SEM)"
    )
    parser.add_argument("--benchmark", default=BENCHMARK_PATH)
    parser.add_argument(
        "--skip-populate",
        action="store_true",
        help="Não recriar índice se exp_voyage-multilingual já existir",
    )
    parser.add_argument("--output-tex", default=OUTPUT_TEX)
    args = parser.parse_args()

    faiss_path = os.path.join(DB_DIR, "faiss_index", "index.faiss")
    if not args.skip_populate and not os.path.exists(faiss_path):
        print(f"Índice não encontrado em {DB_DIR}. Populando...")
        populate_databases(
            embedding_key=EMBEDDING_KEY,
            output_directory=DB_DIR,
        )
    elif not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"Índice Voyage não encontrado em {DB_DIR}. "
            "Rode sem --skip-populate ou execute db_populator."
        )
    else:
        print(f"Usando índice existente: {DB_DIR}")

    benchmark = load_benchmark(args.benchmark)
    print(f"Benchmark: {len(benchmark)} perguntas | Embedding: {EMBEDDING_KEY}\n")

    results = run_fusion_experiment(benchmark)
    save_results(results)

    latex = results_to_latex_sem(results)
    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Tabela LaTeX: {args.output_tex}\n")
    print(latex)


if __name__ == "__main__":
    main()
