"""
Configuração central do projeto: fonte única de paths e valores default.

Em vez de repetir constantes (embedding, chunk size, top-k, paths de dados)
em cada módulo, importe daqui. Assim, mudar um default acontece em um só lugar.

Convenção:
- DEFAULT_*  -> baseline de indexação/retrieval (usado por db_populator,
               database_loader, doc_loader e a grade experimental).
- BEST_*     -> configuração ótima pós-experimentos (voyage + RRF + ablação),
               usada pelo LLM-as-Judge e como referência de produção.
"""

import os

# Raiz do projeto (este arquivo está em src/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------------------------------------
# Paths de dados
# ----------------------------------------------------------------------------
DATA_DIR = "./data"
RAW_PDF_DIR = "./data/raw_IPR"
VECTOR_DB_DIR = "./data/vector_db"
EVAL_DIR = "./data/eval"
BENCHMARK_PATH = "./data/eval/synthetic_benchmark.json"
RESULTS_DIR = "./data/eval/results"
ANALYSIS_DIR = "./data/eval/analysis"
FIG_DIR = "./data/eval/analysis/figures"

# Saídas do LLM-as-Judge
NO_RAG_JSON = "./data/eval/llm_judge_no_rag.json"
FULL_CONTEXT_JSON = "./data/eval/llm_judge_full_context.json"
RAG_JSON = "./data/eval/llm_judge_results.json"

# ----------------------------------------------------------------------------
# Defaults de indexação / retrieval (baseline)
# ----------------------------------------------------------------------------
DEFAULT_EMBEDDING = "openai-large"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150
DEFAULT_TOP_K = 3
DEFAULT_SPARSE_WEIGHT = 0.5
DEFAULT_STRATEGY = "hybrid_rrf"

# RRF
RRF_K = 60
RRF_FETCH_MULTIPLIER = 3

# ----------------------------------------------------------------------------
# Configuração ótima pós-experimentos (LLM-as-Judge / produção)
# ----------------------------------------------------------------------------
BEST_EMBEDDING = "voyage-multilingual"
BEST_STRATEGY = "hybrid_rrf"
BEST_CHUNK_SIZE = 1000
BEST_OVERLAP = 256
BEST_TOP_K = 3
BEST_DB_DIR = "./data/vector_db/exp_chunk_voyage_cs1000_ov256"

# ----------------------------------------------------------------------------
# Modelos de embedding disponíveis
# ----------------------------------------------------------------------------
EMBEDDING_CONFIGS = {
    "openai-small": {"provider": "openai", "model": "text-embedding-3-small"},
    "openai-large": {"provider": "openai", "model": "text-embedding-3-large"},
    "voyage-multilingual": {"provider": "voyage", "model": "voyage-multilingual-2"},
}

# ----------------------------------------------------------------------------
# Métricas / avaliação
# ----------------------------------------------------------------------------
METRICS = ["hit_rate", "mrr", "context_precision", "context_recall"]
RELEVANCE_THRESHOLD = 0.5  # τ: fração mínima de tokens do trecho-fonte no chunk

# ----------------------------------------------------------------------------
# LLM / custo (GPT-4o)
# ----------------------------------------------------------------------------
ENCODING = "o200k_base"
GPT4O_INPUT_USD_PER_1M = 2.50
GPT4O_OUTPUT_USD_PER_1M = 10.00
DEFAULT_TPM_LIMIT = 30_000
