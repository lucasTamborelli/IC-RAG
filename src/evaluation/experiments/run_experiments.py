"""
Orquestrador da grade experimental completa.

Roda todos os experimentos definidos (embedding models, estratégias de fusão,
parâmetros de chunking) contra o benchmark sintético, coleta métricas de retrieval
e salva resultados consolidados.

Uso:
    python -m src.evaluation.experiments.run_experiments
    python -m src.evaluation.experiments.run_experiments --experiments embedding_comparison
    python -m src.evaluation.experiments.run_experiments --experiments chunking_ablation
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.db_populator import populate_databases, EMBEDDING_CONFIGS
from src.retrieval.database_loader import load_databases
from src.retrieval.searchs import (
    semantic_search,
    keyword_search,
    hybrid_search,
    hybrid_search_rrf,
)
from src.evaluation.metrics.retrieval_metrics import evaluate_retrieval, aggregate_metrics
from src.config import (
    BENCHMARK_PATH,
    DEFAULT_SPARSE_WEIGHT,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    RESULTS_DIR,
)


def load_benchmark(path: str = BENCHMARK_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval_experiment(
    faiss_db,
    bm25_retriever,
    benchmark: list[dict],
    strategy: str = DEFAULT_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> dict:
    """
    Roda uma estratégia de retrieval contra o benchmark e retorna métricas agregadas.
    """
    search_fns = {
        "semantic": lambda q: semantic_search(faiss_db, q, top_k=top_k),
        "keyword": lambda q: keyword_search(bm25_retriever, q, top_k=top_k),
        "hybrid_weighted": lambda q: hybrid_search(
            faiss_db, bm25_retriever, q, sparse_weight=sparse_weight, top_k=top_k
        ),
        "hybrid_rrf": lambda q: hybrid_search_rrf(
            faiss_db, bm25_retriever, q, top_k=top_k
        ),
    }

    search_fn = search_fns[strategy]
    per_query_results = []

    for qa in benchmark:
        chunks = search_fn(qa["pergunta"])
        metrics = evaluate_retrieval(chunks, qa["trecho_fonte"])
        metrics["pergunta_id"] = qa["id"]
        metrics["tipo"] = qa.get("tipo", "unknown")
        per_query_results.append(metrics)

    metric_keys = ["hit_rate", "mrr", "context_precision", "context_recall"]
    aggregated = aggregate_metrics([{k: r[k] for k in metric_keys} for r in per_query_results])

    by_type = {}
    for r in per_query_results:
        tipo = r["tipo"]
        by_type.setdefault(tipo, []).append({k: r[k] for k in metric_keys})
    aggregated_by_type = {t: aggregate_metrics(rs) for t, rs in by_type.items()}

    return {
        "aggregated": aggregated,
        "by_type": aggregated_by_type,
        "per_query": per_query_results,
    }


def experiment_embedding_comparison(benchmark: list[dict]) -> list[dict]:
    """
    Experimento 3a: Compara diferentes modelos de embedding.
    Cada modelo é testado com as 4 estratégias de retrieval.
    """
    results = []
    strategies = ["semantic", "keyword", "hybrid_weighted", "hybrid_rrf"]

    for emb_key in EMBEDDING_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Embedding: {emb_key}")
        print(f"{'='*60}")

        db_dir = f"./data/vector_db/exp_{emb_key}"

        try:
            populate_databases(
                embedding_key=emb_key,
                output_directory=db_dir,
            )
        except Exception as e:
            print(f"  [ERRO] Falha ao popular DB com {emb_key}: {e}")
            continue

        for strategy in strategies:
            print(f"  Estratégia: {strategy}...")
            try:
                faiss_db, bm25_retriever = load_databases(db_dir, emb_key)
                result = run_retrieval_experiment(
                    faiss_db, bm25_retriever, benchmark, strategy=strategy,
                )
                results.append({
                    "experiment": "embedding_comparison",
                    "embedding": emb_key,
                    "strategy": strategy,
                    "chunk_size": 1000,
                    "overlap": 150,
                    "top_k": 3,
                    **result["aggregated"],
                    "by_type": result["by_type"],
                    "per_query": result["per_query"],
                })
                print(f"    Hit Rate: {result['aggregated']['hit_rate']['mean']:.3f} | "
                      f"MRR: {result['aggregated']['mrr']['mean']:.3f}")
            except Exception as e:
                print(f"    [ERRO] {strategy}: {e}")

    return results


def experiment_fusion_comparison(benchmark: list[dict]) -> list[dict]:
    """
    Experimento 3b: Compara weighted fusion (vários pesos) vs RRF.
    Usa o embedding padrão (openai-large).
    """
    results = []
    faiss_db, bm25_retriever = load_databases()

    sparse_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    for sw in sparse_weights:
        print(f"  Hybrid weighted (sparse_weight={sw})...")
        result = run_retrieval_experiment(
            faiss_db, bm25_retriever, benchmark,
            strategy="hybrid_weighted", sparse_weight=sw,
        )
        results.append({
            "experiment": "fusion_comparison",
            "strategy": f"hybrid_weighted_sw{sw}",
            "sparse_weight": sw,
            **result["aggregated"],
        })
        print(f"    Hit Rate: {result['aggregated']['hit_rate']['mean']:.3f} | "
              f"MRR: {result['aggregated']['mrr']['mean']:.3f}")

    print(f"  Hybrid RRF...")
    result = run_retrieval_experiment(
        faiss_db, bm25_retriever, benchmark, strategy="hybrid_rrf",
    )
    results.append({
        "experiment": "fusion_comparison",
        "strategy": "hybrid_rrf",
        "sparse_weight": None,
        **result["aggregated"],
    })
    print(f"    Hit Rate: {result['aggregated']['hit_rate']['mean']:.3f} | "
          f"MRR: {result['aggregated']['mrr']['mean']:.3f}")

    return results


def experiment_chunking_ablation(benchmark: list[dict]) -> list[dict]:
    """
    Experimento 3c: Ablação de chunk_size, overlap e top_k.
    Usa hybrid_rrf (melhor fusão esperada) e embedding padrão.
    """
    results = []

    configs = [
        {"chunk_size": 256,  "overlap": 32,  "top_k": 3},
        {"chunk_size": 512,  "overlap": 64,  "top_k": 3},
        {"chunk_size": 1000, "overlap": 150, "top_k": 3},
        {"chunk_size": 2048, "overlap": 256, "top_k": 3},
        {"chunk_size": 1000, "overlap": 0,   "top_k": 3},
        {"chunk_size": 1000, "overlap": 300, "top_k": 3},
        {"chunk_size": 1000, "overlap": 150, "top_k": 2},
        {"chunk_size": 1000, "overlap": 150, "top_k": 5},
        {"chunk_size": 1000, "overlap": 150, "top_k": 10},
    ]

    for cfg in configs:
        label = f"cs{cfg['chunk_size']}_ov{cfg['overlap']}_k{cfg['top_k']}"
        print(f"\n  Config: {label}")

        db_dir = f"./data/vector_db/exp_chunk_{label}"
        populate_databases(
            output_directory=db_dir,
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"],
        )

        faiss_db, bm25_retriever = load_databases(db_dir)
        result = run_retrieval_experiment(
            faiss_db, bm25_retriever, benchmark,
            strategy="hybrid_rrf", top_k=cfg["top_k"],
        )
        results.append({
            "experiment": "chunking_ablation",
            "strategy": "hybrid_rrf",
            **cfg,
            **result["aggregated"],
        })
        print(f"    Hit Rate: {result['aggregated']['hit_rate']['mean']:.3f} | "
              f"MRR: {result['aggregated']['mrr']['mean']:.3f}")

    return results


def save_results(results: list[dict], experiment_name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{experiment_name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {path}")
    return path


EXPERIMENTS = {
    "embedding_comparison": experiment_embedding_comparison,
    "fusion_comparison": experiment_fusion_comparison,
    "chunking_ablation": experiment_chunking_ablation,
}


def main():
    parser = argparse.ArgumentParser(description="Roda grade experimental")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--benchmark", default=BENCHMARK_PATH)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    print(f"Benchmark carregado: {len(benchmark)} perguntas")

    to_run = list(EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments

    all_results = []
    for exp_name in to_run:
        print(f"\n{'#'*60}")
        print(f"# Experimento: {exp_name}")
        print(f"{'#'*60}")
        start = time.time()
        results = EXPERIMENTS[exp_name](benchmark)
        elapsed = time.time() - start
        print(f"\n  Tempo: {elapsed:.1f}s | {len(results)} configurações testadas")
        save_results(results, exp_name)
        all_results.extend(results)

    save_results(all_results, "all_experiments")
    print(f"\nTotal: {len(all_results)} resultados")


if __name__ == "__main__":
    main()
