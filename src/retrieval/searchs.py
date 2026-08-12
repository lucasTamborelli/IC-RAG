from collections import defaultdict
from dotenv import load_dotenv
from typing import List

from src.config import (
        DEFAULT_SPARSE_WEIGHT,
        DEFAULT_TOP_K,
        RRF_FETCH_MULTIPLIER,
        RRF_K,
)


def semantic_search(faiss_db, query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
        results = faiss_db.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

def keyword_search(bm25_retriever, query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
        """
        BM25 com pré-processamento em português.
        O retriever foi construído com preprocess_func=preprocess_pt,
        então a query também precisa passar pelo mesmo pré-processamento.
        """
        bm25_retriever.k = top_k
        # BM25Retriever.invoke() chama internamente preprocess_func na query
        results = bm25_retriever.invoke(query)
        return [doc.page_content for doc in results]

def hybrid_search(faiss_db, bm25_retriever, query: str, sparse_weight: float = DEFAULT_SPARSE_WEIGHT, top_k: int = DEFAULT_TOP_K) -> List[str]:
        sem_scored = faiss_db.similarity_search_with_score(query, k=top_k)
        bm25_retriever.k = top_k
        key_results = bm25_retriever.invoke(query)

        final_scores = defaultdict(lambda: {"text": "", "dense": 0.0, "sparse": 0.0})

        for doc, distance in sem_scored:
                text = doc.page_content
                final_scores[text]["text"] = text
                # Converte distância L2 em score (quanto menor distância, maior score)
                final_scores[text]["dense"] = 1 / (1 + distance)

        for rank, doc in enumerate(key_results):
                text = doc.page_content
                final_scores[text]["text"] = text
                # Score posicional: 1º lugar = 1.0, 2º = (k-1)/k, etc.
                final_scores[text]["sparse"] = (top_k - rank) / top_k

        final = []
        for text, vals in final_scores.items():
                combined = sparse_weight * vals["sparse"] + (1 - sparse_weight) * vals["dense"]
                final.append({"text": vals["text"], "score": combined})

        final.sort(key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in final[:top_k]]


def hybrid_search_rrf(
        faiss_db,
        bm25_retriever,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = RRF_K,
        fetch_multiplier: int = RRF_FETCH_MULTIPLIER,
) -> List[str]:
        """
        Hybrid search usando Reciprocal Rank Fusion (Cormack et al., 2009).
        RRF é mais robusto que média ponderada porque normaliza automaticamente
        as escalas diferentes entre scores densos e esparsos.
        score(d) = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)
        """
        fetch_k = top_k * fetch_multiplier

        sem_results = faiss_db.similarity_search_with_score(query, k=fetch_k)
        bm25_retriever.k = fetch_k
        key_results = bm25_retriever.invoke(query)

        rrf_scores = defaultdict(lambda: {"text": "", "score": 0.0})

        for rank, (doc, _distance) in enumerate(sem_results):
                text = doc.page_content
                rrf_scores[text]["text"] = text
                rrf_scores[text]["score"] += 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(key_results):
                text = doc.page_content
                rrf_scores[text]["text"] = text
                rrf_scores[text]["score"] += 1.0 / (rrf_k + rank + 1)

        final = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in final[:top_k]]