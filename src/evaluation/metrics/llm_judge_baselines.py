"""
Baselines LLM-as-Judge: sem RAG e com contexto completo (full-doc).

Compara qualidade e custo (tokens in/out) contra o pipeline RAG ótimo:
  - no_rag:       pergunta → GPT-4o → judge (contexto vazio)
  - full_context: pergunta + documentos no prompt → GPT-4o → judge
  - rag:          scores/tokens estimados a partir de llm_judge_results.json

ATENÇÃO — limite TPM da API OpenAI (tier 1 comum: 30k TPM):
  Enviar os 7 PDFs juntos (~50k tokens) excede o TPM por requisição.
  Use --context-scope oracle_doc (default) para injetar só o PDF-fonte de cada
  pergunta (~5–14k tokens, cabe no tier 1). Para all_docs, é necessário subir
  o tier da organização (TPM ≥ 60k) ou usar a Batch API.

Saídas:
  - data/eval/llm_judge_no_rag.json
  - data/eval/llm_judge_full_context.json
  - data/eval/analysis/llm_judge_baseline_no_rag.tex
  - data/eval/analysis/llm_judge_baseline_full_context.tex
  - data/eval/analysis/llm_judge_baseline_comparison.tex

Uso:
    python -m src.evaluation.metrics.llm_judge_baselines --mode no_rag
    python -m src.evaluation.metrics.llm_judge_baselines --mode full_context
    python -m src.evaluation.metrics.llm_judge_baselines --mode full_context --context-scope oracle_doc
    python -m src.evaluation.metrics.llm_judge_baselines --mode full_context --context-scope all_docs --tpm-limit 80000
    python -m src.evaluation.metrics.llm_judge_baselines --mode latex
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.evaluation.metrics.llm_judge import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_DIR,
    DEFAULT_EMBEDDING,
    DEFAULT_OVERLAP,
    DEFAULT_STRATEGY,
    DEFAULT_TOP_K,
    JUDGE_PROMPT,
    judge_response,
    resolve_db_dir,
)
from src.ingestion.doc_loader import Treater
from src.llm.model import LLM_cloud
from src.config import (
    ANALYSIS_DIR as OUTPUT_DIR,
    BENCHMARK_PATH,
    DEFAULT_TPM_LIMIT,
    ENCODING,
    FULL_CONTEXT_JSON,
    GPT4O_INPUT_USD_PER_1M,
    GPT4O_OUTPUT_USD_PER_1M,
    NO_RAG_JSON,
    RAG_JSON,
    RAW_PDF_DIR as PDF_DIR,
)

load_dotenv()

PROMPT_OVERHEAD_TOKENS = 250

TYPE_ORDER = ["factual", "multi-hop", "comparativa", "procedimental"]
TYPE_LABELS = {
    "factual": "Factual",
    "multi-hop": "Multi-hop",
    "comparativa": "Comparativa",
    "procedimental": "Procedimental",
}

NO_RAG_GENERATION_PROMPT = """\
Você é um assistente especializado em documentos de Propriedade Intelectual do ITA.

Responda à pergunta abaixo usando apenas seu conhecimento geral.
Se não tiver certeza ou a resposta depender de documentos institucionais específicos do ITA,
diga explicitamente que não possui acesso aos documentos e não pode confirmar.

Pergunta:
{pergunta}
"""


def _enc():
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


def token_pair(prompt: str, completion: str) -> dict[str, int]:
    return {
        "input": count_tokens(prompt),
        "output": count_tokens(completion),
    }


def usage_from_response(response, prompt: str, completion: str) -> dict[str, int]:
    """Prefere metadados da API; fallback tiktoken."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if usage:
        return {
            "input": usage.get("prompt_tokens") or usage.get("input_tokens") or count_tokens(prompt),
            "output": usage.get("completion_tokens") or usage.get("output_tokens") or count_tokens(completion),
        }
    return token_pair(prompt, completion)


def sum_token_stats(rows: list[dict], prefix: str = "tokens") -> dict:
    gen_in = gen_out = judge_in = judge_out = 0
    for r in rows:
        t = r.get(prefix, {})
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
        "estimated_cost_usd": (
            total_in * GPT4O_INPUT_USD_PER_1M + total_out * GPT4O_OUTPUT_USD_PER_1M
        ) / 1_000_000,
    }


def load_benchmark(path: str = BENCHMARK_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_document_text(pdf_dir: str, documento: str) -> tuple[str, dict]:
    """Carrega um único PDF pelo nome do arquivo (campo documento do benchmark)."""
    path = os.path.join(pdf_dir, documento)
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF não encontrado: {path}")

    docs = Treater(path).load_documents()
    text = "\n".join(d.page_content for d in docs)
    meta = {
        "documento": documento,
        "chars": len(text),
        "words": len(text.split()),
        "tokens": count_tokens(text),
    }
    return f"=== {documento} ===\n{text}", meta


def load_full_corpus(pdf_dir: str = PDF_DIR) -> tuple[str, dict]:
    """Concatena texto de todos os PDFs. Retorna (texto, metadados)."""
    paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"Nenhum PDF em {pdf_dir}")

    sections = []
    per_doc = {}
    for path in paths:
        name = os.path.basename(path)
        docs = Treater(path).load_documents()
        text = "\n".join(d.page_content for d in docs)
        sections.append(f"=== {name} ===\n{text}")
        per_doc[name] = {
            "chars": len(text),
            "words": len(text.split()),
            "tokens": count_tokens(text),
        }

    corpus = "\n\n".join(sections)
    meta = {
        "n_documents": len(paths),
        "total_chars": len(corpus),
        "total_words": len(corpus.split()),
        "total_tokens": count_tokens(corpus),
        "per_document": per_doc,
    }
    return corpus, meta


def resolve_context_for_question(
    qa: dict,
    pdf_dir: str,
    context_scope: str,
    corpus_cache: tuple[str, dict] | None = None,
) -> tuple[str, dict, str]:
    """
    Retorna (texto_contexto, meta, scope_label).

    context_scope:
      - all_docs:    todos os 7 PDFs (~50k tokens) — requer TPM alto
      - oracle_doc:  só o PDF indicado em qa['documento'] (~5–14k tokens)
    """
    if context_scope == "all_docs":
        if corpus_cache is None:
            corpus_cache = load_full_corpus(pdf_dir)
        return corpus_cache[0], corpus_cache[1], "all_docs"

    if context_scope == "oracle_doc":
        documento = qa.get("documento")
        if not documento:
            raise ValueError(f"Pergunta {qa.get('id')} sem campo 'documento' no benchmark.")
        text, meta = load_document_text(pdf_dir, documento)
        return text, meta, "oracle_doc"

    raise ValueError(f"context_scope inválido: {context_scope}")


def preflight_tpm(prompt: str, tpm_limit: int, step: str) -> int:
    """
    Verifica se o prompt cabe no TPM da organização antes de chamar a API.
    Retorna contagem de tokens; levanta ValueError se exceder.
    """
    n = count_tokens(prompt)
    budget = tpm_limit - PROMPT_OVERHEAD_TOKENS
    if n > budget:
        raise ValueError(
            f"[{step}] Prompt com ~{n:,} tokens excede o TPM disponível "
            f"(limite {tpm_limit:,}, margem {budget:,}).\n"
            f"  → Use --context-scope oracle_doc (cabe no tier 1), ou\n"
            f"  → Aumente o tier OpenAI (https://platform.openai.com/account/limits), ou\n"
            f"  → Passe --tpm-limit <seu_novo_limite> se já tiver upgrade."
        )
    return n


def _sleep_between_calls(delay_seconds: float):
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def generation_prompt_no_rag(pergunta: str) -> str:
    return NO_RAG_GENERATION_PROMPT.format(pergunta=pergunta)


def generation_prompt_with_context(pergunta: str, contexto: str) -> str:
    llm = LLM_cloud(model="gpt-4o", temperature=0.1)
    return llm.prompt(pergunta, contexto)


def judge_prompt_text(
    pergunta: str,
    resposta: str,
    contexto: str,
    resposta_referencia: str,
) -> str:
    return JUDGE_PROMPT.format(
        pergunta=pergunta,
        resposta=resposta,
        contexto=contexto or "(nenhum contexto fornecido ao sistema)",
        resposta_referencia=resposta_referencia,
    )


def generate_answer(
    llm: ChatOpenAI,
    prompt: str,
    tpm_limit: int | None = None,
    step: str = "generation",
) -> tuple[str, dict[str, int]]:
    if tpm_limit is not None:
        preflight_tpm(prompt, tpm_limit, step)
    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content.strip()
    tokens = usage_from_response(response, prompt, content)
    return content, tokens


def evaluate_with_judge(
    judge_llm: ChatOpenAI,
    pergunta: str,
    resposta: str,
    contexto: str,
    resposta_referencia: str,
    tpm_limit: int | None = None,
) -> tuple[dict, dict[str, int]]:
    prompt = judge_prompt_text(pergunta, resposta, contexto, resposta_referencia)
    if tpm_limit is not None:
        preflight_tpm(prompt, tpm_limit, "judge")
    response = judge_llm.invoke([{"role": "user", "content": prompt}])
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]

    tokens = usage_from_response(response, prompt, content)

    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        scores = judge_response(
            judge_llm, pergunta, resposta, contexto, resposta_referencia
        )
        tokens = token_pair(prompt, json.dumps(scores, ensure_ascii=False))

    return scores, tokens


def run_no_rag(
    benchmark_path: str = BENCHMARK_PATH,
    output_path: str = NO_RAG_JSON,
    model: str = "gpt-4o",
) -> list[dict]:
    benchmark = load_benchmark(benchmark_path)
    gen_llm = ChatOpenAI(model=model, temperature=0.1)
    judge_llm = ChatOpenAI(model=model, temperature=0.0)

    results = []
    for i, qa in enumerate(benchmark):
        print(f"  [no_rag {i+1}/{len(benchmark)}] {qa['id']}")

        gen_prompt = generation_prompt_no_rag(qa["pergunta"])
        resposta, gen_tokens = generate_answer(gen_llm, gen_prompt)

        scores, judge_tokens = evaluate_with_judge(
            judge_llm,
            pergunta=qa["pergunta"],
            resposta=resposta,
            contexto="",
            resposta_referencia=qa["resposta_referencia"],
        )

        results.append({
            "id": qa["id"],
            "tipo": qa.get("tipo"),
            "mode": "no_rag",
            "pergunta": qa["pergunta"],
            "resposta": resposta,
            "resposta_referencia": qa["resposta_referencia"],
            "contexto_tokens": 0,
            "tokens": {
                "generation": gen_tokens,
                "judge": judge_tokens,
                "total": {
                    "input": gen_tokens["input"] + judge_tokens["input"],
                    "output": gen_tokens["output"] + judge_tokens["output"],
                },
            },
            **{k: scores.get(k) for k in (
                "faithfulness", "answer_relevancy", "correctness",
                "faithfulness_justificativa", "answer_relevancy_justificativa",
                "correctness_justificativa",
            )},
        })

    _save_results(results, output_path, mode="no_rag")
    return results


def run_full_context(
    benchmark_path: str = BENCHMARK_PATH,
    output_path: str = FULL_CONTEXT_JSON,
    pdf_dir: str = PDF_DIR,
    model: str = "gpt-4o",
    context_scope: str = "oracle_doc",
    tpm_limit: int = DEFAULT_TPM_LIMIT,
    delay_seconds: float = 0.0,
    resume: bool = True,
):
    benchmark = load_benchmark(benchmark_path)
    corpus_cache = load_full_corpus(pdf_dir) if context_scope == "all_docs" else None

    gen_llm = ChatOpenAI(model=model, temperature=0.1)
    judge_llm = ChatOpenAI(model=model, temperature=0.0)

    if context_scope == "all_docs":
        print(
            f"  Escopo: todos os PDFs — {corpus_cache[1]['total_tokens']:,} tokens "
            f"(requer TPM ≥ {corpus_cache[1]['total_tokens'] + 2000:,})"
        )
    else:
        print("  Escopo: oracle_doc — 1 PDF-fonte por pergunta (campo 'documento' do benchmark)")

    existing: dict[str, dict] = {}
    if resume and os.path.exists(output_path):
        try:
            for row in _load_result_rows(output_path):
                if row.get("faithfulness") is not None:
                    existing[row["id"]] = row
            if existing:
                print(f"  Retomando: {len(existing)}/{len(benchmark)} perguntas já concluídas")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    results = []
    for i, qa in enumerate(benchmark):
        if qa["id"] in existing:
            results.append(existing[qa["id"]])
            continue

        contexto, ctx_meta, scope_label = resolve_context_for_question(
            qa, pdf_dir, context_scope, corpus_cache
        )
        ctx_tokens = count_tokens(contexto)
        print(
            f"  [full_context {i+1}/{len(benchmark)}] {qa['id']} "
            f"({scope_label}, ~{ctx_tokens:,} tok)"
        )

        gen_prompt = generation_prompt_with_context(qa["pergunta"], contexto)
        resposta, gen_tokens = generate_answer(
            gen_llm, gen_prompt, tpm_limit=tpm_limit, step="generation"
        )
        _sleep_between_calls(delay_seconds)

        scores, judge_tokens = evaluate_with_judge(
            judge_llm,
            pergunta=qa["pergunta"],
            resposta=resposta,
            contexto=contexto,
            resposta_referencia=qa["resposta_referencia"],
            tpm_limit=tpm_limit,
        )
        _sleep_between_calls(delay_seconds)

        row = {
            "id": qa["id"],
            "tipo": qa.get("tipo"),
            "mode": "full_context",
            "context_scope": scope_label,
            "documento": qa.get("documento"),
            "pergunta": qa["pergunta"],
            "resposta": resposta,
            "resposta_referencia": qa["resposta_referencia"],
            "contexto_tokens": ctx_tokens,
            "context_meta": ctx_meta,
            "tokens": {
                "generation": gen_tokens,
                "judge": judge_tokens,
                "total": {
                    "input": gen_tokens["input"] + judge_tokens["input"],
                    "output": gen_tokens["output"] + judge_tokens["output"],
                },
            },
            **{k: scores.get(k) for k in (
                "faithfulness", "answer_relevancy", "correctness",
                "faithfulness_justificativa", "answer_relevancy_justificativa",
                "correctness_justificativa",
            )},
        }
        results.append(row)
        existing[qa["id"]] = row
        _save_results(results, output_path, mode=f"full_context_{scope_label}")

    extra_meta = corpus_cache[1] if corpus_cache else {"context_scope": context_scope}
    _save_results(results, output_path, mode=f"full_context_{context_scope}", extra_meta=extra_meta)
    return results


def estimate_rag_tokens(
    rag_results: list[dict],
    benchmark_path: str = BENCHMARK_PATH,
    embedding_key: str = DEFAULT_EMBEDDING,
    search_strategy: str = DEFAULT_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    db_dir: str = DEFAULT_DB_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Estima tokens do RAG retroativamente (sem chamar API):
    re-recupera chunks e reconstrói prompts de geração/judge.
    """
    from src.retrieval.database_loader import load_databases
    from src.retrieval.searchs import (
        hybrid_search_rrf,
        hybrid_search,
        keyword_search,
        semantic_search,
    )

    benchmark = {q["id"]: q for q in load_benchmark(benchmark_path)}
    resolved_db = resolve_db_dir(db_dir, chunk_size, overlap, top_k)
    faiss_db, bm25_retriever = load_databases(resolved_db, embedding_key)

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

    enriched = []
    for r in rag_results:
        qa = benchmark[r["id"]]
        chunks = search_fn(qa["pergunta"])
        contexto = "\n\n---\n\n".join(chunks)
        resposta = r.get("resposta_rag") or r.get("resposta", "")

        gen_prompt = generation_prompt_with_context(qa["pergunta"], contexto)
        judge_prompt = judge_prompt_text(
            qa["pergunta"], resposta, contexto, qa["resposta_referencia"]
        )
        gen_tokens = token_pair(gen_prompt, resposta)
        judge_tokens = token_pair(judge_prompt, json.dumps({
            "faithfulness": r.get("faithfulness"),
            "answer_relevancy": r.get("answer_relevancy"),
            "correctness": r.get("correctness"),
        }))

        enriched.append({
            **r,
            "mode": "rag",
            "contexto_tokens": count_tokens(contexto),
            "tokens": {
                "generation": gen_tokens,
                "judge": judge_tokens,
                "total": {
                    "input": gen_tokens["input"] + judge_tokens["input"],
                    "output": gen_tokens["output"] + judge_tokens["output"],
                },
            },
        })
    return enriched


def _save_results(
    results: list[dict],
    output_path: str,
    mode: str,
    extra_meta: dict | None = None,
):
    valid = [r for r in results if r.get("faithfulness") is not None]
    summary = {
        "mode": mode,
        "n_questions": len(results),
        "scores": _avg_scores(valid),
        "tokens": sum_token_stats(results),
    }
    if extra_meta:
        summary["corpus"] = extra_meta

    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    tok = summary["tokens"]
    print(f"\n  [{mode}] Faith={summary['scores']['faithfulness']:.2f} "
          f"Rel={summary['scores']['answer_relevancy']:.2f} "
          f"Corr={summary['scores']['correctness']:.2f}")
    print(f"  Tokens totais: in={tok['total_input']:,} out={tok['total_output']:,} "
          f"(~US$ {tok['estimated_cost_usd']:.2f})")
    print(f"  Salvo em: {output_path}")


def _avg_scores(rows: list[dict]) -> dict[str, float]:
    dims = ["faithfulness", "answer_relevancy", "correctness"]
    out = {}
    for d in dims:
        vals = [r[d] for r in rows if r.get(d) is not None]
        out[d] = sum(vals) / len(vals) if vals else 0.0
    return out


def _avg_scores_by_type(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_type: dict[str, list[dict]] = {t: [] for t in TYPE_ORDER}
    for r in rows:
        tipo = r.get("tipo", "")
        if tipo in by_type:
            by_type[tipo].append(r)

    result = {}
    for tipo, rs in by_type.items():
        result[tipo] = _avg_scores(rs) if rs else {
            "faithfulness": 0.0, "answer_relevancy": 0.0, "correctness": 0.0,
        }
    result["overall"] = _avg_scores(rows)
    return result


def _fmt_score(x: float) -> str:
    return f"{x:.2f}"


def _fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def generate_baseline_table_latex(
    rows: list[dict],
    mode_label: str,
    caption_extra: str,
    label: str,
) -> str:
    by_type = _avg_scores_by_type(rows)
    tok = sum_token_stats(rows)

    body = []
    for tipo in TYPE_ORDER:
        s = by_type[tipo]
        body.append(
            f"{TYPE_LABELS[tipo]} & {_fmt_score(s['faithfulness'])} & "
            f"{_fmt_score(s['answer_relevancy'])} & {_fmt_score(s['correctness'])} \\\\"
        )
    o = by_type["overall"]
    body.append("\\midrule")
    body.append(
        f"\\textbf{{Média geral}} & \\textbf{{{_fmt_score(o['faithfulness'])}}} & "
        f"\\textbf{{{_fmt_score(o['answer_relevancy'])}}} & "
        f"\\textbf{{{_fmt_score(o['correctness'])}}} \\\\"
    )
    body.append("\\midrule")
    body.append(
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Tokens (28 perguntas):}} "
        f"geração in={_fmt_int(tok['generation_input'])}, out={_fmt_int(tok['generation_output'])}; "
        f"judge in={_fmt_int(tok['judge_input'])}, out={_fmt_int(tok['judge_output'])}; "
        f"total in={_fmt_int(tok['total_input'])}, out={_fmt_int(tok['total_output'])} "
        f"(~US\\$ {tok['estimated_cost_usd']:.2f})}} \\\\"
    )
    body.append(
        f"\\multicolumn{{4}}{{l}}{{\\textit{{Média por pergunta:}} "
        f"in={_fmt_int(tok['avg_input_per_question'])}, "
        f"out={_fmt_int(tok['avg_output_per_question'])}}} \\\\"
    )

    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{Avaliação LLM-as-Judge — {mode_label} (escala 1--5, $n=28$; {caption_extra})}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Tipo & Faithfulness & Answer Relevancy & Correctness \\\\\n\\midrule\n"
        + "\n".join(body) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


def generate_comparison_table_latex(
    no_rag: list[dict],
    full_context: list[dict],
    rag: list[dict],
) -> str:
    configs = [
        ("Sem RAG", no_rag, "apenas pergunta, sem documentos"),
        ("Full-doc no prompt", full_context, "todos os PDFs no contexto"),
        (
            "RAG ótimo",
            rag,
            "Voyage + hybrid RRF, cs=1000, ov=256, $k=3$",
        ),
    ]

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Comparação de baselines LLM-as-Judge: qualidade vs.\\ custo em tokens ($n=28$)}",
        "\\label{tab:llm_judge_baselines}",
        "\\small",
        "\\begin{tabular}{lcccrrrrrr}",
        "\\toprule",
        "Config & Faith. & Rel. & Corr. & Gen in & Gen out & Judge in & Judge out & Total in & Total out \\\\",
        "\\midrule",
    ]

    for name, rows, _ in configs:
        scores = _avg_scores([r for r in rows if r.get("faithfulness") is not None])
        tok = sum_token_stats(rows)
        lines.append(
            f"{name} & {_fmt_score(scores['faithfulness'])} & "
            f"{_fmt_score(scores['answer_relevancy'])} & "
            f"{_fmt_score(scores['correctness'])} & "
            f"{_fmt_int(tok['generation_input'])} & {_fmt_int(tok['generation_output'])} & "
            f"{_fmt_int(tok['judge_input'])} & {_fmt_int(tok['judge_output'])} & "
            f"\\textbf{{{_fmt_int(tok['total_input'])}}} & {_fmt_int(tok['total_output'])} \\\\"
        )

    lines.extend([
        "\\midrule",
        f"\\multicolumn{{10}}{{l}}{{\\textit{{Custo estimado GPT-4o (input US\\$ {GPT4O_INPUT_USD_PER_1M:.2f}/M, "
        f"output US\\$ {GPT4O_OUTPUT_USD_PER_1M:.2f}/M):}} "
        f"Sem RAG ~US\\$ {sum_token_stats(no_rag)['estimated_cost_usd']:.2f}; "
        f"Full-doc ~US\\$ {sum_token_stats(full_context)['estimated_cost_usd']:.2f}; "
        f"RAG ~US\\$ {sum_token_stats(rag)['estimated_cost_usd']:.2f}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_latex_tables(
    no_rag_path: str = NO_RAG_JSON,
    full_context_path: str = FULL_CONTEXT_JSON,
    rag_path: str = RAG_JSON,
    output_dir: str = OUTPUT_DIR,
    estimate_rag: bool = True,
):
    no_rag = _load_result_rows(no_rag_path)
    full_context = _load_result_rows(full_context_path)

    if estimate_rag and os.path.exists(rag_path):
        with open(rag_path, "r", encoding="utf-8") as f:
            rag_raw = json.load(f)
        rag_rows = rag_raw if isinstance(rag_raw, list) else rag_raw.get("results", rag_raw)
        rag = estimate_rag_tokens(rag_rows)
    else:
        rag = _load_result_rows(rag_path) if os.path.exists(rag_path) else []

    os.makedirs(output_dir, exist_ok=True)

    tables = {
        "llm_judge_baseline_no_rag.tex": generate_baseline_table_latex(
            no_rag,
            "baseline sem RAG",
            "GPT-4o responde sem acesso aos documentos",
            "tab:llm_judge_no_rag",
        ),
        "llm_judge_baseline_full_context.tex": generate_baseline_table_latex(
            full_context,
            "baseline full-doc",
            "GPT-4o com todos os PDFs transcritos no prompt",
            "tab:llm_judge_full_context",
        ),
        "llm_judge_baseline_comparison.tex": generate_comparison_table_latex(
            no_rag, full_context, rag
        ),
    }

    for filename, latex in tables.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"  LaTeX: {path}")

    return tables


def _load_result_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def main():
    parser = argparse.ArgumentParser(
        description="Baselines LLM-as-Judge (sem RAG e full-doc) com contagem de tokens"
    )
    parser.add_argument(
        "--mode",
        choices=["no_rag", "full_context", "latex", "all"],
        required=True,
        help="no_rag/full_context: roda experimento; latex: só gera tabelas; all: roda ambos + latex",
    )
    parser.add_argument("--benchmark", default=BENCHMARK_PATH)
    parser.add_argument("--pdf-dir", default=PDF_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-rag-output", default=NO_RAG_JSON)
    parser.add_argument("--full-context-output", default=FULL_CONTEXT_JSON)
    parser.add_argument("--rag-results", default=RAG_JSON)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--context-scope",
        choices=["oracle_doc", "all_docs"],
        default="oracle_doc",
        help="oracle_doc: 1 PDF-fonte/pergunta (~15k tok, cabe no tier 1); "
             "all_docs: 7 PDFs (~50k tok, requer TPM alto)",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=DEFAULT_TPM_LIMIT,
        help=f"TPM da organização OpenAI para preflight (default: {DEFAULT_TPM_LIMIT})",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Pausa entre chamadas gen/judge (útil se TPM acumula por minuto; ex.: 65)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Não retomar full_context a partir de JSON parcial existente",
    )
    args = parser.parse_args()

    if args.mode in ("no_rag", "all"):
        print("=== Baseline: sem RAG ===")
        run_no_rag(args.benchmark, args.no_rag_output, args.model)

    if args.mode in ("full_context", "all"):
        print("\n=== Baseline: full-doc no prompt ===")
        run_full_context(
            args.benchmark,
            args.full_context_output,
            args.pdf_dir,
            args.model,
            context_scope=args.context_scope,
            tpm_limit=args.tpm_limit,
            delay_seconds=args.delay_seconds,
            resume=not args.no_resume,
        )

    if args.mode in ("latex", "all"):
        print("\n=== Gerando tabelas LaTeX ===")
        generate_latex_tables(
            no_rag_path=args.no_rag_output,
            full_context_path=args.full_context_output,
            rag_path=args.rag_results,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
