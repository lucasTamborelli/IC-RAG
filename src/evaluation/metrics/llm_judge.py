"""
LLM-as-Judge: avaliação automatizada de respostas RAG usando GPT-4o.

Avalia três dimensões com rubrica estruturada:
1. Faithfulness (fidelidade ao contexto recuperado)
2. Answer Relevancy (relevância da resposta à pergunta)
3. Correctness (corretude vs. resposta de referência)

Referência: Zheng et al., 2023 — "Judging LLM-as-a-Judge"
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json

import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import (
    BENCHMARK_PATH,
    ENCODING,
    GPT4O_INPUT_USD_PER_1M,
    GPT4O_OUTPUT_USD_PER_1M,
    RAG_JSON,
    VECTOR_DB_DIR,
)
from src.config import BEST_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
from src.config import BEST_DB_DIR as DEFAULT_DB_DIR
from src.config import BEST_EMBEDDING as DEFAULT_EMBEDDING
from src.config import BEST_OVERLAP as DEFAULT_OVERLAP
from src.config import BEST_STRATEGY as DEFAULT_STRATEGY
from src.config import BEST_TOP_K as DEFAULT_TOP_K

load_dotenv()

JUDGE_PROMPT = """Você é um avaliador especialista de sistemas de pergunta-resposta sobre documentos institucionais.

Avalie a resposta abaixo em duas dimensões, usando a rubrica fornecida.

## Pergunta
{pergunta}

## Resposta do sistema
{resposta}

## Contexto recuperado (chunks usados para gerar a resposta)
{contexto}

## Resposta de referência (ground truth)
{resposta_referencia}

---

## Rubrica de avaliação

### Faithfulness (1-5): A resposta é suportada pelo contexto recuperado?
1 — Resposta contém informações fabricadas ou contraditórias ao contexto
2 — Resposta parcialmente suportada, com afirmações sem base no contexto
3 — Resposta majoritariamente suportada, com detalhes menores sem base
4 — Resposta bem suportada pelo contexto, com no máximo 1 detalhe inferido
5 — Resposta inteiramente suportada por informações explícitas no contexto

### Answer Relevancy (1-5): A resposta aborda de fato a pergunta?
1 — Resposta irrelevante ou sobre outro assunto
2 — Resposta tangencialmente relacionada, não responde a pergunta
3 — Resposta parcialmente relevante, aborda a pergunta mas de forma incompleta
4 — Resposta relevante e quase completa, faltando apenas detalhes menores
5 — Resposta diretamente relevante e completa em relação à pergunta

### Correctness (1-5): A resposta está correta em relação à resposta de referência?
1 — Resposta completamente incorreta
2 — Resposta com erros graves
3 — Resposta parcialmente correta
4 — Resposta correta com imprecisões menores
5 — Resposta correta e alinhada com a referência

## Formato de saída (JSON):
Retorne APENAS um JSON válido, sem markdown:
{{
  "faithfulness": <1-5>,
  "faithfulness_justificativa": "justificativa breve",
  "answer_relevancy": <1-5>,
  "answer_relevancy_justificativa": "justificativa breve",
  "correctness": <1-5>,
  "correctness_justificativa": "justificativa breve"
}}
"""


def _enc():
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


def usage_from_response(response, prompt: str, completion: str) -> dict[str, int]:
    """Prefere metadados da API; fallback tiktoken."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if usage:
        return {
            "input": (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or count_tokens(prompt)
            ),
            "output": (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or count_tokens(completion)
            ),
        }
    return {"input": count_tokens(prompt), "output": count_tokens(completion)}


def sum_token_stats(rows: list[dict]) -> dict:
    gen_in = gen_out = judge_in = judge_out = 0
    for r in rows:
        t = r.get("tokens", {})
        gen_in += t.get("generation", {}).get("input", 0)
        gen_out += t.get("generation", {}).get("output", 0)
        judge_in += t.get("judge", {}).get("input", 0)
        judge_out += t.get("judge", {}).get("output", 0)
    total_in = gen_in + judge_in
    total_out = gen_out + judge_out
    n = max(len(rows), 1)
    return {
        "generation_input": gen_in,
        "generation_output": gen_out,
        "judge_input": judge_in,
        "judge_output": judge_out,
        "total_input": total_in,
        "total_output": total_out,
        "avg_input_per_question": total_in / n,
        "avg_output_per_question": total_out / n,
        "avg_context_tokens": sum(r.get("contexto_tokens", 0) for r in rows) / n,
        "estimated_cost_usd": (
            total_in * GPT4O_INPUT_USD_PER_1M + total_out * GPT4O_OUTPUT_USD_PER_1M
        ) / 1_000_000,
    }


def resolve_db_dir(
    db_dir: str,
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> str:
    """Resolve diretório do índice, incluindo nomes legados com _k{top_k}."""
    candidates = [
        db_dir,
        f"{VECTOR_DB_DIR}/exp_chunk_voyage_cs{chunk_size}_ov{overlap}",
        f"{VECTOR_DB_DIR}/exp_chunk_voyage_cs{chunk_size}_ov{overlap}_k{top_k}",
    ]
    for path in candidates:
        faiss_path = os.path.join(path, "faiss_index", "index.faiss")
        if os.path.exists(faiss_path):
            return path
    return db_dir


def judge_prompt_text(
    pergunta: str,
    resposta: str,
    contexto: str,
    resposta_referencia: str,
) -> str:
    return JUDGE_PROMPT.format(
        pergunta=pergunta,
        resposta=resposta,
        contexto=contexto,
        resposta_referencia=resposta_referencia,
    )


def _parse_judge_json(content: str) -> dict:
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"  [WARN] JSON inválido do judge: {content[:200]}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "correctness": None,
            "error": "json_parse_failed",
        }


def judge_with_tokens(
    llm: ChatOpenAI,
    pergunta: str,
    resposta: str,
    contexto: str,
    resposta_referencia: str,
) -> tuple[dict, dict[str, int]]:
    """Avalia resposta e retorna (scores, tokens)."""
    prompt = judge_prompt_text(pergunta, resposta, contexto, resposta_referencia)
    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content.strip()
    tokens = usage_from_response(response, prompt, content)
    return _parse_judge_json(content), tokens


def judge_response(
    llm: ChatOpenAI,
    pergunta: str,
    resposta: str,
    contexto: str,
    resposta_referencia: str,
) -> dict:
    """Avalia uma resposta com rubrica estruturada usando LLM-as-Judge."""
    scores, _ = judge_with_tokens(
        llm, pergunta, resposta, contexto, resposta_referencia
    )
    return scores


def generate_rag_answer(
    rag_llm,
    pergunta: str,
    contexto: str,
) -> tuple[str, dict[str, int]]:
    """Gera resposta RAG e retorna (texto, tokens)."""
    prompt = rag_llm.prompt(pergunta, contexto)
    response = rag_llm.model.invoke([{"role": "user", "content": prompt}])
    content = response.content.strip()
    tokens = usage_from_response(response, prompt, content)
    return content, tokens


def run_llm_judge(
    benchmark_path: str = BENCHMARK_PATH,
    output_path: str = RAG_JSON,
    model: str = "gpt-4o",
    search_strategy: str = DEFAULT_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    embedding_key: str = DEFAULT_EMBEDDING,
    db_dir: str = DEFAULT_DB_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Pipeline completo: carrega benchmark, gera respostas RAG, avalia com LLM-as-Judge.
    Registra tokens de input/output por pergunta (geração + judge) e summary agregado.
    """
    from src.retrieval.database_loader import load_databases
    from src.retrieval.searchs import (
        semantic_search, keyword_search, hybrid_search, hybrid_search_rrf,
    )
    from src.llm.model import LLM_cloud

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    resolved_db = resolve_db_dir(db_dir, chunk_size, overlap, top_k)
    faiss_path = os.path.join(resolved_db, "faiss_index", "index.faiss")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"Índice não encontrado em {resolved_db}. "
            f"Rode: python -m src.evaluation.experiments.run_chunking_experiments "
            f"(config cs={chunk_size}, ov={overlap})"
        )

    print(
        f"Retrieval: {embedding_key} + {search_strategy} | "
        f"cs={chunk_size}, ov={overlap}, k={top_k} | DB: {resolved_db}"
    )

    faiss_db, bm25_retriever = load_databases(resolved_db, embedding_key)
    rag_llm = LLM_cloud(model="gpt-4o", temperature=0.1)
    judge_llm = ChatOpenAI(model=model, temperature=0.0)

    search_fns = {
        "semantic": lambda q: semantic_search(faiss_db, q, top_k=top_k),
        "keyword": lambda q: keyword_search(bm25_retriever, q, top_k=top_k),
        "hybrid_weighted": lambda q: hybrid_search(
            faiss_db, bm25_retriever, q, top_k=top_k
        ),
        "hybrid_rrf": lambda q: hybrid_search_rrf(
            faiss_db, bm25_retriever, q, top_k=top_k
        ),
    }
    search_fn = search_fns[search_strategy]

    config_meta = {
        "embedding": embedding_key,
        "strategy": search_strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "db_dir": resolved_db,
    }

    results = []
    for i, qa in enumerate(benchmark):
        print(f"  [{i+1}/{len(benchmark)}] {qa['id']}: {qa['pergunta'][:60]}...")

        chunks = search_fn(qa["pergunta"])
        contexto = "\n\n---\n\n".join(chunks)
        contexto_tokens = count_tokens(contexto)

        resposta, gen_tokens = generate_rag_answer(rag_llm, qa["pergunta"], contexto)
        scores, judge_tokens = judge_with_tokens(
            judge_llm,
            pergunta=qa["pergunta"],
            resposta=resposta,
            contexto=contexto,
            resposta_referencia=qa["resposta_referencia"],
        )

        results.append({
            "id": qa["id"],
            "tipo": qa.get("tipo"),
            "pergunta": qa["pergunta"],
            "resposta_rag": resposta,
            "resposta_referencia": qa["resposta_referencia"],
            "n_chunks": len(chunks),
            "contexto_tokens": contexto_tokens,
            "tokens": {
                "generation": gen_tokens,
                "judge": judge_tokens,
                "total": {
                    "input": gen_tokens["input"] + judge_tokens["input"],
                    "output": gen_tokens["output"] + judge_tokens["output"],
                },
            },
            **config_meta,
            **scores,
        })

    valid = [r for r in results if r.get("faithfulness") is not None]
    if valid:
        avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
        avg_relev = sum(r["answer_relevancy"] for r in valid) / len(valid)
        avg_corr = sum(r["correctness"] for r in valid) / len(valid)
        print(f"\nMédia Faithfulness: {avg_faith:.2f}")
        print(f"Média Answer Relevancy: {avg_relev:.2f}")
        print(f"Média Correctness: {avg_corr:.2f}")

    token_summary = sum_token_stats(results)
    print(
        f"\nTokens (28 perguntas): "
        f"gen in={token_summary['generation_input']:,} out={token_summary['generation_output']:,} | "
        f"judge in={token_summary['judge_input']:,} out={token_summary['judge_output']:,} | "
        f"total in={token_summary['total_input']:,} out={token_summary['total_output']:,}"
    )
    print(
        f"Média/pergunta: in={token_summary['avg_input_per_question']:,.0f} "
        f"out={token_summary['avg_output_per_question']:,.0f} | "
        f"contexto recuperado={token_summary['avg_context_tokens']:,.0f} tok"
    )
    print(f"Custo estimado GPT-4o: ~US$ {token_summary['estimated_cost_usd']:.2f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        choices=["semantic", "keyword", "hybrid_weighted", "hybrid_rrf"])
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--benchmark", default=BENCHMARK_PATH)
    parser.add_argument("--output", default=RAG_JSON)
    args = parser.parse_args()

    run_llm_judge(
        benchmark_path=args.benchmark,
        output_path=args.output,
        search_strategy=args.strategy,
        top_k=args.top_k,
        embedding_key=args.embedding,
        db_dir=args.db_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
