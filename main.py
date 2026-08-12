from src.ingestion.doc_loader import *
from src.llm.model import *
from src.retrieval.database_loader import *
from src.retrieval.preprocess import *
from src.retrieval.aux_helpers import *
from src.retrieval.searchs import *
from src.ingestion.db_populator import *
import streamlit as sl
import json


if __name__ == "__main__":
        
        llm = LLM_cloud(model="gpt-4o", temperature = 0.1)
        
        try: faiss_db, bm25_retriever = load_databases()
        except Exception as e: sl.error(f"Erro ao carregar databases: {e}")

        if 'avaliacoes' not in sl.session_state:
                sl.session_state.avaliacoes = []
        if 'respostas' not in sl.session_state: 
                sl.session_state.respostas = {"Semantic": None, "Keyword": None, "Hybrid": None}
        if 'ultima_query' not in sl.session_state:
                sl.session_state.ultima_query = ""
        if 'tokens' not in sl.session_state:
                sl.session_state.tokens = {"Semantic": [0, 0], "Keyword": [0, 0], "Hybrid": [0, 0]}
        if 'eval_round' not in sl.session_state:
                sl.session_state.eval_round = 0
        if 'observacao' not in sl.session_state:
                sl.session_state.observacao = ""

        query = sl.text_input(label='Pergunta:')
        buscar = sl.button('Buscar') 
        if buscar or (query and query != sl.session_state.ultima_query):
                
                sem_chunks = semantic_search(faiss_db, query, top_k=3)
                key_chunks = keyword_search(bm25_retriever, query, top_k=3)
                hyb_chunks = hybrid_search(faiss_db, bm25_retriever, query, sparse_weight=0.5, top_k=3)
                
                ctx_semantico = "\n\n".join(sem_chunks)
                ctx_keyword = "\n\n".join(key_chunks)
                ctx_hibrido = "\n\n".join(hyb_chunks)
                
                sl.session_state.respostas['Semantic'] = llm.response(query, ctx_semantico)
                sl.session_state.tokens['Semantic'] = n_tokens(llm.prompt(query, ctx_semantico), sl.session_state.respostas['Semantic'], "o200k_base")

                sl.session_state.respostas['Keyword'] = llm.response(query, ctx_keyword)
                sl.session_state.tokens['Keyword'] = n_tokens(llm.prompt(query, ctx_keyword), sl.session_state.respostas['Keyword'], "o200k_base")

                sl.session_state.respostas['Hybrid'] = llm.response(query, ctx_hibrido)
                sl.session_state.tokens['Hybrid'] = n_tokens(llm.prompt(query, ctx_hibrido), sl.session_state.respostas['Hybrid'], "o200k_base")
                sl.session_state.ultima_query = query
                sl.session_state.observacao = ""
                sl.session_state.eval_round += 1
                
        tab1, tab2, tab3 = sl.tabs(['Semantic', 'Keyword', 'Hybrid'])   
                
        with tab1:
                render_tab("Semantic", sl.session_state.respostas['Semantic'], sl.session_state.tokens['Semantic']) 
        with tab2:
                render_tab("Keyword", sl.session_state.respostas['Keyword'], sl.session_state.tokens['Keyword'])
        with tab3:
                render_tab("Hybrid", sl.session_state.respostas['Hybrid'], sl.session_state.tokens['Hybrid'])

        sl.write("---")
        sl.markdown("### Avaliações")
        opcoes = ("1 (Muito Ruim)", "2", "3", "4", "5 (Muito Bom)")
        rnd = sl.session_state.eval_round

        col1, col2, col3 = sl.columns(3)
        with col1:
                eval_sem = sl.radio("Semantic:", opcoes, key=f"eval_Semantic_{rnd}", index=None)
        with col2:
                eval_key = sl.radio("Keyword:", opcoes, key=f"eval_Keyword_{rnd}", index=None)
        with col3:
                eval_hyb = sl.radio("Hybrid:", opcoes, key=f"eval_Hybrid_{rnd}", index=None)

        sl.write("---")
        observacao = sl.text_area(
                "Observações sobre as respostas:",
                value=sl.session_state.observacao,
                key=f"obs_{rnd}",
                placeholder="Escreva aqui o que achou das respostas..."
        )
        sl.session_state.observacao = observacao

        enviar = sl.button("Enviar avaliações")
        if enviar:
                todas_preenchidas = all([eval_sem, eval_key, eval_hyb])
                if not todas_preenchidas:
                        sl.warning("Preencha as 3 avaliações antes de enviar.")
                else:
                        for method, score in [("Semantic", eval_sem), ("Keyword", eval_key), ("Hybrid", eval_hyb)]:
                                tk = sl.session_state.tokens[method]
                                sl.session_state.avaliacoes.append({
                                        "Pergunta": sl.session_state.ultima_query,
                                        "Tipo": method,
                                        "Resposta": sl.session_state.respostas[method],
                                        "Avaliacao": score,
                                        "Tokens_input": tk[0],
                                        "Tokens_output": tk[1],
                                        "Observacao": sl.session_state.observacao
                                })
                        sl.session_state.eval_round += 1
                        sl.success("Avaliações registradas!")
                        sl.rerun()

        with sl.sidebar:
                sl.markdown("### Exportar avaliações")
                sl.write(f"Avaliações registradas: {len(sl.session_state.avaliacoes)}")
                if sl.session_state.avaliacoes:
                        sl.download_button(
                                label="Baixar avaliações (.json)",
                                data=json.dumps(sl.session_state.avaliacoes, ensure_ascii=False, indent=2),
                                file_name="avaliacoes.json",
                                mime="application/json"
                        )
                