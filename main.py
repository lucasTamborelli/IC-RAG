from classes import *
from local_functions import *
import streamlit as sl
import json
from db_populator import *

if __name__ == "__main__":
        
        llm = LLM_cloud(model="gpt-4o", temperature = 0.1)
        
        try: faiss_db, bm25_retriever = load_databases()
        except Exception as e: sl.error("Did not find databases")

        if 'avaliacoes' not in sl.session_state:
                sl.session_state.avaliacoes = []
        if 'avaliacoes_salvas' not in sl.session_state:
                sl.session_state.avaliacoes_salvas = set()
        if 'respostas' not in sl.session_state: 
                sl.session_state.respostas = {"Semantic": None, "Keyword": None, "Hybrid": None}
        if 'ultima_query' not in sl.session_state:
                sl.session_state.ultima_query = ""
        if 'tokens' not in sl.session_state:
                sl.session_state.tokens = {"Semantic": [0, 0], "Keyword": [0, 0], "Hybrid": [0, 0]}

        query = sl.text_input(label='Pergunta:')
        buscar = sl.button('Buscar') 
        if buscar or (query and query != sl.session_state.ultima_query):
                
                sem_chunks = semantic_search(faiss_db, query, top_k=3)
                key_chunks = keyword_search(bm25_retriever, query, top_k=3)
                hyb_chunks = hybrid_search(faiss_db, bm25_retriever, query, top_k=3)
                
                
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

                for method in ["Semantic", "Keyword", "Hybrid"]:
                        sl.session_state.pop(f'eval_{method}', None)
                        sl.session_state.pop(f'avaliacao_{method}', None)
                
        tab1, tab2, tab3 = sl.tabs(['Semantic', 'Keyword', 'Hybrid'])   
                
        with tab1:
                render_tab(query, "Semantic", sl.session_state.respostas['Semantic'], sl.session_state.tokens['Semantic']) 
        with tab2:
                render_tab(query, "Keyword", sl.session_state.respostas['Keyword'], sl.session_state.tokens['Keyword'])
        with tab3:
                render_tab(query, "Hybrid", sl.session_state.respostas['Hybrid'], sl.session_state.tokens['Hybrid'])

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
                