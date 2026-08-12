"""
Métricas de qualidade de retrieval para avaliação automática.

Todas as métricas são calculadas comparando os chunks recuperados
com o trecho_fonte do benchmark sintético.

Métricas implementadas:
- Hit Rate @k: pelo menos 1 chunk contém o trecho-fonte
- MRR (Mean Reciprocal Rank): posição do primeiro chunk relevante
- Context Precision @k: fração dos k chunks que são relevantes
- Context Recall: cobertura do trecho-fonte pelos chunks recuperados
"""

from __future__ import annotations
import unicodedata
import re

from src.config import RELEVANCE_THRESHOLD


def _normalize(text: str) -> str:
    """Normaliza texto para comparação: lowercase, sem acentos, sem pontuação extra."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_set(text: str) -> set[str]:
    return set(_normalize(text).split())


def _is_relevant(chunk: str, trecho_fonte: str, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """
    Um chunk é relevante se a sobreposição de tokens com o trecho-fonte
    excede o threshold (fração dos tokens do trecho que aparecem no chunk).
    """
    fonte_tokens = _token_set(trecho_fonte)
    if not fonte_tokens:
        return False
    chunk_tokens = _token_set(chunk)
    overlap = len(fonte_tokens & chunk_tokens) / len(fonte_tokens)
    return overlap >= threshold


def hit_rate(
    retrieved_chunks: list[str],
    trecho_fonte: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """1.0 se pelo menos um chunk é relevante, 0.0 caso contrário."""
    for chunk in retrieved_chunks:
        if _is_relevant(chunk, trecho_fonte, threshold):
            return 1.0
    return 0.0


def mrr(
    retrieved_chunks: list[str],
    trecho_fonte: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """1/posição do primeiro chunk relevante (1-indexed). 0.0 se nenhum é relevante."""
    for i, chunk in enumerate(retrieved_chunks):
        if _is_relevant(chunk, trecho_fonte, threshold):
            return 1.0 / (i + 1)
    return 0.0


def context_precision(
    retrieved_chunks: list[str],
    trecho_fonte: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Fração dos chunks recuperados que são relevantes."""
    if not retrieved_chunks:
        return 0.0
    relevant = sum(
        1 for c in retrieved_chunks if _is_relevant(c, trecho_fonte, threshold)
    )
    return relevant / len(retrieved_chunks)


def context_recall(
    retrieved_chunks: list[str],
    trecho_fonte: str,
) -> float:
    """Fração dos tokens do trecho-fonte cobertos pela união dos chunks recuperados."""
    fonte_tokens = _token_set(trecho_fonte)
    if not fonte_tokens:
        return 0.0
    all_chunk_tokens = set()
    for chunk in retrieved_chunks:
        all_chunk_tokens |= _token_set(chunk)
    return len(fonte_tokens & all_chunk_tokens) / len(fonte_tokens)


def evaluate_retrieval(
    retrieved_chunks: list[str],
    trecho_fonte: str,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """Calcula todas as métricas para uma única query."""
    return {
        "hit_rate": hit_rate(retrieved_chunks, trecho_fonte, threshold),
        "mrr": mrr(retrieved_chunks, trecho_fonte, threshold),
        "context_precision": context_precision(retrieved_chunks, trecho_fonte, threshold),
        "context_recall": context_recall(retrieved_chunks, trecho_fonte),
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Agrega métricas de múltiplas queries.
    Retorna média e desvio-padrão para cada métrica.
    """
    if not results:
        return {}

    metrics = list(results[0].keys())
    agg = {}
    for m in metrics:
        values = [r[m] for r in results]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5
        agg[m] = {"mean": round(mean, 4), "std": round(std, 4)}
    return agg
