# Avaliação de Estratégias de Retrieval em RAG para Documentos Institucionais em Português

Projeto de Iniciação Científica (PIBIC/ENCITA-ITA) que avalia experimentalmente estratégias de *retrieval* num *pipeline* **RAG** (*Retrieval-Augmented Generation*) aplicado aos documentos da Pró-Reitoria de Pesquisa e Relacionamento Institucional do ITA (**IPR/ITA**).

O objetivo é responder: **qual combinação de modelo de *embedding*, estratégia de fusão e parâmetros de *chunking* maximiza a qualidade da recuperação** em documentos normativos em português?

## Principais resultados

- Melhor configuração: **`voyage-multilingual-2` + hybrid RRF** (Hit Rate = 0,929; MRR = 0,816), com *chunk size* = 1000 e top-*k* = 3.
- *Embeddings* multilíngues superam modelos gerais maiores: `text-embedding-3-small` superou o `text-embedding-3-large` no corpus em português.
- Na avaliação *end-to-end* (LLM-as-Judge), o RAG atingiu **Correctness 4,39/5,0**, contra **1,82/5,0** sem contexto documental.
- Perguntas procedimentais são o ponto fraco residual (Correctness 3,43/5,0).

## Arquitetura

O sistema divide-se em duas fases: **indexação *offline*** dos PDFs e **consulta *online*** que recupera trechos e gera a resposta.

```
                    ┌─────────────────── Fase OFFLINE (indexação) ───────────────────┐
  PDFs (IPR/ITA) ─► Tratamento/extração ─► Segmentação em chunks ─┬─► FAISS  (denso / embeddings)
                    (PyMuPDF)              (size + overlap)        └─► BM25   (esparso / preprocess PT)

                    ┌─────────────────── Fase ONLINE (consulta) ─────────────────────┐
  Pergunta ─► Seleção da estratégia ─► [semantic | keyword | hybrid_weighted | hybrid_rrf]
           ─► Montagem do contexto (top-k chunks) ─► Geração GPT-4o ─► Resposta + tokens
```

**Estratégias de recuperação:** semântica (FAISS), *keyword* (BM25), híbrida ponderada (`score = w·sparse + (1-w)·dense`) e híbrida RRF (*Reciprocal Rank Fusion*, κ = 60).

## Estrutura do projeto

```
cod/
├── main.py                     # App Streamlit de inspeção comparativa das estratégias
├── requirements.txt
├── data/
│   ├── raw_IPR/                # PDFs fonte (7 documentos IPR/ITA)
│   ├── vector_db/              # Índices FAISS + BM25 serializados
│   └── eval/                   # Benchmark, resultados e tabelas/figuras
└── src/
    ├── config.py               # Fonte única de defaults e paths
    ├── ingestion/
    │   ├── doc_loader.py       # Carrega PDF (PyMuPDF) e faz chunking
    │   └── db_populator.py     # Popula FAISS + BM25
    ├── llm/model.py            # Wrapper ChatOpenAI (GPT-4o)
    ├── retrieval/
    │   ├── searchs.py          # semantic, keyword, hybrid_weighted, hybrid_rrf
    │   ├── database_loader.py  # Carrega os índices
    │   └── preprocess.py       # Tokenização/stopwords em português (BM25)
    └── evaluation/
        ├── metrics/            # retrieval_metrics, llm_judge, llm_judge_baselines
        ├── experiments/        # benchmark_generator, run_*_experiments
        └── viz/                # analysis, generate_charts, generate_poster
```

## Como executar

```bash
# 1. Dependências
pip install -r requirements.txt

# 2. Variáveis de ambiente (.env)
#    OPENAI_API_KEY=sk-...
#    VOYAGE_API_KEY=...        # opcional, para voyage-multilingual-2

# 3. Indexar os PDFs (parametrizável por embedding, chunk size e overlap)
python -m src.ingestion.db_populator
python -m src.ingestion.db_populator --embedding voyage-multilingual --chunk-size 512 --overlap 64

# 4. Gerar o benchmark sintético (28 perguntas categorizadas)
python -m src.evaluation.experiments.benchmark_generator

# 5. Rodar a grade experimental
python -m src.evaluation.experiments.run_experiments

# 6. Avaliação end-to-end (LLM-as-Judge)
python -m src.evaluation.metrics.llm_judge --strategy hybrid_rrf

# 7. Análise + tabelas LaTeX e figuras
python -m src.evaluation.viz.analysis

# App de inspeção manual (Streamlit)
streamlit run main.py
```

> No Windows, defina `KMP_DUPLICATE_LIB_OK=TRUE` antes de rodar os experimentos (conflito libomp do FAISS).

## Avaliação

- **Métricas de retrieval:** Hit Rate @k, MRR, Context Precision @k e Context Recall. Um *chunk* é relevante quando cobre ≥ 50% dos *tokens* do trecho-fonte.
- **LLM-as-Judge:** GPT-4o (T = 0,0) avalia *Faithfulness*, *Answer Relevancy* e *Correctness* (escala 1–5), comparando o RAG ótimo a um *baseline* sem contexto e a um teto teórico (*oracle-doc*).

## Referências

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR.
- Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.

---

Autor: Lucas Guedes Tamborelli — Iniciação Científica, ITA.
