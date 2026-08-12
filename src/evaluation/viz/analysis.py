"""
Análise estatística e geração de tabelas/figuras para o artigo.

Carrega os resultados dos experimentos e produz:
1. Tabelas LaTeX formatadas para o artigo
2. Figuras (heatmaps, bar charts) em PNG/PDF
3. Testes estatísticos (Wilcoxon signed-rank para pares)

Uso:
    python -m src.evaluation.viz.analysis
    python -m src.evaluation.viz.analysis --results-dir data/eval/results
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.config import ANALYSIS_DIR as OUTPUT_DIR, METRICS, RESULTS_DIR


def load_latest_results(results_dir: str, experiment_name: str) -> list[dict]:
    """Carrega o arquivo de resultados mais recente para um experimento."""
    files = [
        f for f in os.listdir(results_dir)
        if f.startswith(experiment_name) and f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(f"Nenhum resultado para '{experiment_name}' em {results_dir}")
    files.sort(reverse=True)
    path = os.path.join(results_dir, files[0])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def results_to_table(results: list[dict], row_key: str, columns: list[str] = None) -> str:
    """Converte resultados em tabela formatada para terminal."""
    if columns is None:
        columns = METRICS

    header = f"{'Config':<35}" + "".join(f"{m:>18}" for m in columns)
    lines = [header, "-" * len(header)]

    for r in results:
        row_label = str(r.get(row_key, "?"))[:35]
        cells = []
        for m in columns:
            val = r.get(m, {})
            if isinstance(val, dict):
                mean = val.get("mean", 0)
                std = val.get("std", 0)
                cells.append(f"{mean:.3f} ± {std:.3f}")
            else:
                cells.append(f"{val:.3f}" if isinstance(val, (int, float)) else str(val))
        lines.append(f"{row_label:<35}" + "".join(f"{c:>18}" for c in cells))

    return "\n".join(lines)


def results_to_latex(
    results: list[dict],
    row_key: str,
    caption: str,
    label: str,
    columns: list[str] = None,
) -> str:
    """Gera tabela LaTeX formatada para o artigo."""
    if columns is None:
        columns = METRICS

    col_headers = {
        "hit_rate": "Hit Rate",
        "mrr": "MRR",
        "context_precision": "Ctx. Prec.",
        "context_recall": "Ctx. Recall",
    }

    n_cols = len(columns) + 1
    col_spec = "l" + "c" * len(columns)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = "Configuração & " + " & ".join(col_headers.get(c, c) for c in columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    best_per_col = {}
    for c in columns:
        values = []
        for r in results:
            val = r.get(c, {})
            if isinstance(val, dict):
                values.append(val.get("mean", 0))
            elif isinstance(val, (int, float)):
                values.append(val)
            else:
                values.append(0)
        if values:
            best_per_col[c] = max(values)

    for r in results:
        row_label = str(r.get(row_key, "?")).replace("_", r"\_")
        cells = [row_label]
        for c in columns:
            val = r.get(c, {})
            if isinstance(val, dict):
                mean = val.get("mean", 0)
                std = val.get("std", 0)
                cell = f"{mean:.3f} ± {std:.3f}"
                if mean == best_per_col.get(c):
                    cell = r"\textbf{" + cell + "}"
            elif isinstance(val, (int, float)):
                cell = f"{val:.3f}"
                if val == best_per_col.get(c):
                    cell = r"\textbf{" + cell + "}"
            else:
                cell = str(val)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


TIPO_LABELS = {
    "factual": "Factual",
    "procedimental": "Procedimental",
    "comparativa": "Comparativa",
    "multi-hop": "Multi-hop",
}

TIPO_ORDER = ["factual", "procedimental", "comparativa", "multi-hop"]

N_PER_TIPO = 7


def _hits_from_mean(mean: float, n: int = N_PER_TIPO) -> int:
    """Converte média de Hit Rate (0/1) em contagem de acertos."""
    return round(mean * n)


def _format_hr_cell(mean: float, std: float, n: int = N_PER_TIPO) -> str:
    hits = _hits_from_mean(mean, n)
    return f"{hits}/{n} ({mean:.2f}±{std:.2f})"


def generate_embedding_by_type_table(
    results_dir: str,
    strategies: list[str] | None = None,
    metric: str = "hit_rate",
) -> tuple[str, str]:
    """
    Tabela de embedding segmentada por tipo de pergunta.
    Mostra acertos/total por tipo (n=7) além da média±std.
    """
    results = load_latest_results(results_dir, "embedding_comparison")
    if strategies is None:
        strategies = ["semantic", "hybrid_rrf"]

    filtered = [
        r for r in results
        if r["strategy"] in strategies and r.get("by_type")
    ]
    filtered.sort(key=lambda x: (x["embedding"], x["strategy"]))

    metric_label = {
        "hit_rate": "Hit Rate @k",
        "mrr": "MRR",
        "context_precision": "Ctx. Prec.",
        "context_recall": "Ctx. Recall",
    }.get(metric, metric)

    tipo_headers = [TIPO_LABELS[t] for t in TIPO_ORDER]

    metric_caption = (
        f"Valores de Hit Rate expressos como acertos/total ($n=7$ por tipo)."
        if metric == "hit_rate"
        else f"Valores de {metric_label} (média por tipo, $n=7$)."
    )

    # --- Terminal ---
    header = f"{'Config':<32}" + "".join(f"{h:>22}" for h in tipo_headers) + f"{'Geral':>22}"
    term_lines = [
        f"Embedding por tipo — {metric_label} (n=7 por tipo, 28 total)",
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    for r in filtered:
        config = f"{r['embedding']} + {r['strategy']}"
        cells = []
        for tipo in TIPO_ORDER:
            val = r["by_type"][tipo][metric]
            if metric == "hit_rate":
                cells.append(_format_hr_cell(val["mean"], val["std"]))
            else:
                cells.append(f"{val['mean']:.3f}±{val['std']:.3f}")
        overall = r[metric]
        if metric == "hit_rate":
            geral = _format_hr_cell(overall["mean"], overall["std"], n=28)
        else:
            geral = f"{overall['mean']:.3f}±{overall['std']:.3f}"
        term_lines.append(
            f"{config:<32}" + "".join(f"{c:>22}" for c in cells) + f"{geral:>22}"
        )

    terminal = "\n".join(term_lines)

    # --- LaTeX ---
    col_spec = "l" + "c" * len(TIPO_ORDER) + "c"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Comparação de embeddings por tipo de pergunta ({metric_label.replace('@k', '@$k$')}). "
        f"{metric_caption}}}",
        r"\label{tab:embedding_by_type_" + metric.replace("_", "") + "}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Configuração & "
        + " & ".join(tipo_headers)
        + r" & Geral \\",
        r"\midrule",
    ]

    for r in filtered:
        config = f"{r['embedding']} + {r['strategy']}".replace("_", r"\_")
        cells = []
        for tipo in TIPO_ORDER:
            val = r["by_type"][tipo][metric]
            if metric == "hit_rate":
                hits = _hits_from_mean(val["mean"])
                cell = f"{hits}/{N_PER_TIPO}"
            else:
                cell = f"{val['mean']:.3f}"
            cells.append(cell)
        if metric == "hit_rate":
            hits_total = _hits_from_mean(r[metric]["mean"], n=28)
            geral = f"{hits_total}/28"
        else:
            geral = f"{r[metric]['mean']:.3f}"
        lines.append(config + " & " + " & ".join(cells) + f" & {geral} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex = "\n".join(lines)

    return terminal, latex


def generate_embedding_comparison_table(results_dir: str) -> tuple[str, str]:
    results = load_latest_results(results_dir, "embedding_comparison")

    pivoted = defaultdict(dict)
    for r in results:
        emb = r["embedding"]
        strategy = r["strategy"]
        key = f"{emb} + {strategy}"
        for m in METRICS:
            pivoted[key][m] = r.get(m, {})
        pivoted[key]["embedding"] = emb
        pivoted[key]["strategy"] = strategy
        pivoted[key]["config"] = key

    rows = list(pivoted.values())
    rows.sort(key=lambda x: (x["embedding"], x["strategy"]))

    terminal = results_to_table(rows, "config")
    latex = results_to_latex(
        rows, "config",
        caption="Comparação de modelos de embedding por estratégia de retrieval",
        label="tab:embedding_comparison",
    )
    return terminal, latex


def generate_fusion_comparison_table(results_dir: str) -> tuple[str, str]:
    results = load_latest_results(results_dir, "fusion_comparison")

    terminal = results_to_table(results, "strategy")
    latex = results_to_latex(
        results, "strategy",
        caption="Comparação de estratégias de fusão hybrid",
        label="tab:fusion_comparison",
    )
    return terminal, latex


def generate_chunking_ablation_table(results_dir: str) -> tuple[str, str]:
    results = load_latest_results(results_dir, "chunking_ablation")

    for r in results:
        r["config"] = f"cs={r['chunk_size']}, ov={r['overlap']}, k={r['top_k']}"

    terminal = results_to_table(results, "config")
    latex = results_to_latex(
        results, "config",
        caption="Ablação de parâmetros de chunking e top-k",
        label="tab:chunking_ablation",
    )
    return terminal, latex


def _fmt_int_tokens(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def _load_judge_results(eval_dir: str) -> list[dict]:
    path = os.path.join(eval_dir, "llm_judge_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError("llm_judge_results.json não encontrado.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def generate_judge_by_type_table(eval_dir: str = "./data/eval") -> tuple[str, str]:
    """Gera tabela LaTeX LLM-as-Judge por tipo de pergunta (com tokens se disponível)."""
    from src.evaluation.metrics.llm_judge import sum_token_stats

    results = _load_judge_results(eval_dir)
    valid = [r for r in results if r.get("faithfulness") is not None]
    dims = ["faithfulness", "answer_relevancy", "correctness"]
    type_order = ["factual", "multi-hop", "comparativa", "procedimental"]
    type_labels = {
        "factual": "Factual",
        "multi-hop": "Multi-hop",
        "comparativa": "Comparativa",
        "procedimental": "Procedimental",
    }

    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("tipo", "?")].append(r)

    def fmt(vals: list[float]) -> str:
        return f"{sum(vals) / len(vals):.2f}" if vals else "---"

    body_rows = []
    col_avgs = [[] for _ in dims]
    for tipo in type_order:
        rs = by_type.get(tipo, [])
        cells = []
        for i, d in enumerate(dims):
            vals = [r[d] for r in rs if r.get(d) is not None]
            col_avgs[i].extend(vals)
            cells.append(fmt(vals))
        body_rows.append(f"{type_labels[tipo]} & {' & '.join(cells)} \\\\")

    overall = [fmt(v) for v in col_avgs]
    body_rows.append(
        f"\\textbf{{Média geral}} & \\textbf{{{overall[0]}}} & "
        f"\\textbf{{{overall[1]}}} & \\textbf{{{overall[2]}}} \\\\"
    )

    if valid and valid[0].get("tokens"):
        tok = sum_token_stats(valid)
        body_rows.append("\\midrule")
        body_rows.append(
            f"\\multicolumn{{4}}{{l}}{{\\textit{{Tokens (28 perguntas):}} "
            f"geração in={_fmt_int_tokens(tok['generation_input'])}, "
            f"out={_fmt_int_tokens(tok['generation_output'])}; "
            f"judge in={_fmt_int_tokens(tok['judge_input'])}, "
            f"out={_fmt_int_tokens(tok['judge_output'])}; "
            f"total in={_fmt_int_tokens(tok['total_input'])}, "
            f"out={_fmt_int_tokens(tok['total_output'])} "
            f"(~US\\$ {tok['estimated_cost_usd']:.2f})}} \\\\"
        )
        body_rows.append(
            f"\\multicolumn{{4}}{{l}}{{\\textit{{Média por pergunta:}} "
            f"in={_fmt_int_tokens(tok['avg_input_per_question'])}, "
            f"out={_fmt_int_tokens(tok['avg_output_per_question'])}}} \\\\"
        )

    caption = (
        "Avaliação LLM-as-Judge — RAG ótimo (escala 1--5, $n=28$; "
        "Voyage multilíngue + hybrid RRF, chunk size = 1000, overlap = 256, top-$k$ = 3)"
    )
    latex = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:llm_judge}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Tipo & Faithfulness & Answer Relevancy & Correctness \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    terminal_lines = ["LLM-as-Judge por tipo", "=" * 40]
    for tipo in type_order:
        rs = by_type.get(tipo, [])
        for d in dims:
            vals = [r[d] for r in rs if r.get(d) is not None]
            if vals:
                terminal_lines.append(f"  {tipo} / {d}: {sum(vals)/len(vals):.2f}")
    for i, d in enumerate(dims):
        if col_avgs[i]:
            terminal_lines.append(f"  overall / {d}: {sum(col_avgs[i])/len(col_avgs[i]):.2f}")
    if valid and valid[0].get("tokens"):
        tok = sum_token_stats(valid)
        terminal_lines.append(f"\n  tokens total in={tok['total_input']:,} out={tok['total_output']:,}")
        terminal_lines.append(f"  custo ~US$ {tok['estimated_cost_usd']:.2f}")

    return "\n".join(terminal_lines), latex


def generate_judge_summary(eval_dir: str = "./data/eval") -> str:
    """Gera sumário das avaliações do LLM-as-Judge."""
    path = os.path.join(eval_dir, "llm_judge_results.json")
    if not os.path.exists(path):
        return "Arquivo llm_judge_results.json não encontrado."

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    valid = [r for r in results if r.get("faithfulness") is not None]
    if not valid:
        return "Nenhum resultado válido."

    dims = ["faithfulness", "answer_relevancy", "correctness"]
    lines = ["LLM-as-Judge Summary", "=" * 40]
    for d in dims:
        vals = [r[d] for r in valid if r.get(d) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            lines.append(f"  {d}: {mean:.2f} (n={len(vals)})")

    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("tipo", "?")].append(r)

    lines.append("\nPor tipo de pergunta:")
    for tipo, rs in sorted(by_type.items()):
        for d in dims:
            vals = [r[d] for r in rs if r.get(d) is not None]
            if vals:
                mean = sum(vals) / len(vals)
                lines.append(f"  {tipo} / {d}: {mean:.2f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Análise de resultados experimentais")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generators = {
        "embedding_comparison": generate_embedding_comparison_table,
        "fusion_comparison": generate_fusion_comparison_table,
        "chunking_ablation": generate_chunking_ablation_table,
    }

    by_type_generators = {
        "hit_rate": lambda rd: generate_embedding_by_type_table(rd, metric="hit_rate"),
        "mrr": lambda rd: generate_embedding_by_type_table(rd, metric="mrr"),
    }

    all_latex = []

    for name, gen_fn in generators.items():
        try:
            terminal, latex = gen_fn(args.results_dir)
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            print(terminal)
            all_latex.append(f"% {name}")
            all_latex.append(latex)

            latex_path = os.path.join(args.output_dir, f"{name}.tex")
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"\n  LaTeX salvo em: {latex_path}")
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {name}: {e}")

    for metric_name, gen_fn in by_type_generators.items():
        try:
            terminal, latex = gen_fn(args.results_dir)
            print(f"\n{'='*60}")
            print(f"  embedding_by_type ({metric_name})")
            print(f"{'='*60}")
            print(terminal)

            latex_path = os.path.join(args.output_dir, f"embedding_by_type_{metric_name}.tex")
            with open(latex_path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"\n  LaTeX salvo em: {latex_path}")
            all_latex.append(f"% embedding_by_type_{metric_name}")
            all_latex.append(latex)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] embedding_by_type_{metric_name}: {e}")

    judge_summary = generate_judge_summary()
    print(f"\n{'='*60}")
    print(judge_summary)

    try:
        terminal, latex = generate_judge_by_type_table()
        print(f"\n{'='*60}")
        print("  llm_judge_by_type")
        print(f"{'='*60}")
        print(terminal)
        latex_path = os.path.join(args.output_dir, "llm_judge_by_type.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n  LaTeX salvo em: {latex_path}")
        all_latex.append("% llm_judge_by_type")
        all_latex.append(latex)
    except FileNotFoundError as e:
        print(f"\n  [SKIP] llm_judge_by_type: {e}")

    if all_latex:
        combined_path = os.path.join(args.output_dir, "all_tables.tex")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_latex))
        print(f"\nTodas as tabelas LaTeX salvas em: {combined_path}")


if __name__ == "__main__":
    main()
