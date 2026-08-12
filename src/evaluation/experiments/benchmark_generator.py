"""
Geração automática de benchmark sintético a partir dos PDFs da IPR.

Usa GPT-4o para gerar tuplas (pergunta, resposta, trecho-fonte) categorizados
em 4 tipos: factual, procedimental, comparativa, multi-hop.

Referências:
- RAGAS Testset Generator (Es et al., 2023)
- Automatic QA generation (SQuAD-style)
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.ingestion.doc_loader import Treater
from src.config import BENCHMARK_PATH, RAW_PDF_DIR

load_dotenv()

GENERATION_PROMPT = """Você é um especialista em criação de benchmarks de avaliação para sistemas de busca (retrieval).

Dado o trecho abaixo de um documento institucional de Propriedade Intelectual do ITA, gere exatamente {n_per_type} pergunta(s) de cada um dos 4 tipos abaixo. Cada pergunta deve ser respondível EXCLUSIVAMENTE com base no trecho fornecido.

## Tipos de pergunta:
1. **factual** — Pergunta direta sobre um fato específico mencionado no texto (quem, o quê, quando, onde).
2. **procedimental** — Pergunta sobre um processo, fluxo ou sequência de etapas descrita no texto.
3. **comparativa** — Pergunta que exige comparar ou distinguir dois conceitos, entidades ou processos do texto.
4. **multi-hop** — Pergunta que requer combinar duas ou mais informações diferentes do texto para ser respondida.

## Trecho do documento:
Documento: {documento}
Página(s): {paginas}

---
{texto}
---

## Formato de saída (JSON):
Retorne APENAS um array JSON válido, sem markdown, sem explicações, no formato:
[
  {{
    "tipo": "factual",
    "pergunta": "...",
    "resposta_referencia": "resposta completa baseada no trecho",
    "trecho_fonte": "copie aqui o trecho exato do texto que sustenta a resposta (máx 300 chars)"
  }},
  ...
]

Gere exatamente {total} perguntas ({n_per_type} de cada tipo). Perguntas em português brasileiro.
"""


def load_full_documents(directory: str) -> list[dict]:
    pdf_paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    documents = []
    for path in pdf_paths:
        treater = Treater(path)
        pages = treater.load_documents()
        doc_name = os.path.basename(path)
        for page in pages:
            documents.append({
                "documento": doc_name,
                "pagina": page.metadata.get("page", 0) + 1,
                "texto": page.page_content,
            })
    return documents


def group_pages_into_passages(documents: list[dict], pages_per_group: int = 3) -> list[dict]:
    grouped = []
    by_doc = {}
    for doc in documents:
        by_doc.setdefault(doc["documento"], []).append(doc)

    for doc_name, pages in by_doc.items():
        pages.sort(key=lambda x: x["pagina"])
        for i in range(0, len(pages), pages_per_group):
            group = pages[i:i + pages_per_group]
            texto = "\n\n".join(p["texto"] for p in group)
            if len(texto.strip()) < 200:
                continue
            paginas = f"{group[0]['pagina']}-{group[-1]['pagina']}"
            grouped.append({
                "documento": doc_name,
                "paginas": paginas,
                "texto": texto,
            })
    return grouped


def generate_qa_from_passage(
    llm: ChatOpenAI,
    passage: dict,
    n_per_type: int = 1,
) -> list[dict]:
    total = n_per_type * 4
    prompt = GENERATION_PROMPT.format(
        documento=passage["documento"],
        paginas=passage["paginas"],
        texto=passage["texto"][:6000],
        n_per_type=n_per_type,
        total=total,
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]

    try:
        qa_pairs = json.loads(content)
    except json.JSONDecodeError:
        print(f"  [WARN] JSON invalido para {passage['documento']} p.{passage['paginas']}")
        return []

    for qa in qa_pairs:
        qa["documento"] = passage["documento"]
        qa["pagina"] = passage["paginas"]

    return qa_pairs


def generate_benchmark(
    pdf_directory: str = RAW_PDF_DIR,
    output_path: str = BENCHMARK_PATH,
    target_total: int = 28,
    model: str = "gpt-4o",
) -> list[dict]:
    """
    Pipeline completo: carrega PDFs → agrupa páginas → gera QA → salva JSON.

    Gera ~target_total perguntas distribuídas entre os documentos e tipos.
    """
    llm = ChatOpenAI(model=model, temperature=0.3)

    print("Carregando documentos...")
    documents = load_full_documents(pdf_directory)
    passages = group_pages_into_passages(documents, pages_per_group=3)
    print(f"  {len(documents)} paginas -> {len(passages)} passagens agrupadas")

    n_passages_to_use = min(len(passages), target_total // 4)
    step = max(1, len(passages) // n_passages_to_use)
    selected = passages[::step][:n_passages_to_use]

    print(f"  Usando {len(selected)} passagens para gerar ~{target_total} perguntas")

    all_qa = []
    for i, passage in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {passage['documento']} p.{passage['paginas']}...")
        qa_pairs = generate_qa_from_passage(llm, passage, n_per_type=1)
        all_qa.extend(qa_pairs)

    for idx, qa in enumerate(all_qa):
        qa["id"] = f"q{idx+1:02d}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    tipos = {}
    for qa in all_qa:
        tipos[qa.get("tipo", "?")] = tipos.get(qa.get("tipo", "?"), 0) + 1

    print(f"\nBenchmark gerado: {len(all_qa)} perguntas")
    print(f"  Distribuição: {tipos}")
    print(f"  Salvo em: {output_path}")

    return all_qa


if __name__ == "__main__":
    generate_benchmark()
