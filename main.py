from classes import *
from functions import *
import streamlit as sl
import glob


if __name__ == "__main__":
        
        directory = './IPRdocuments'
        documentList = glob.glob(os.path.join(directory, '*.pdf'))
        
        chunks = []
        
        for archive in documentList:
                document = Treater(archive)
                text = document.extract_text()
                if text:
                        pdf_chunks = document.split_chunks(text, chunk_size = 5000, overlap = 1000)
                        chunks.extend(pdf_chunks)
                        
        llm = LLM_cloud(model="gemini-2.5-pro", temperature = 0.1)


        if 'resposta_llm' not in sl.session_state: 
                sl.session_state.respostas = {"Semantic": None, "Keyword": None, "Hybrid": None}
        if 'ultima_query' not in sl.session_state:
                sl.session_state.ultima_query = ""

        query = sl.text_input(label='Pergunta:')
        buscar = sl.button('Buscar') 
        if buscar or (query and query != sl.session_state.ultima_query):
                semanticCtx = semantic_search('teste', chunks, query, 3, False)
                sl.session_state.respostas['Semantic'] = llm.response(query, semanticCtx)
                semanticTokens = n_tokens(llm.prompt(query, semanticCtx), sl.session_state.respostas['Semantic'], "o200k_harmony")

                keywordCtx = keyword_search('teste22', chunks, query, 3, False)
                sl.session_state.respostas['Keyword'] = llm.response(query, keywordCtx)
                keywordTokens = n_tokens(llm.prompt(query, keywordCtx), sl.session_state.respostas['Keyword'], "o200k_harmony")

                hybridCtx = hybrid_search(keywordCtx, semanticCtx, sparse_weight = 0.6, top_k = 3, printer = False)
                sl.session_state.respostas['Hybrid'] = hybridAnswer = llm.response(query, hybridCtx)
                hybridTokens = n_tokens(llm.prompt(query, hybridCtx), llm.response(query, semanticCtx), "o200k_harmony")

                sl.session_state.ultima_query = query
                
        tab1, tab2, tab3 = sl.tabs(['Semantic', 'Keyword', 'Hybrid'])   
                
        with tab1:
                render_tab(query, "Semantic", sl.session_state.respostas['Semantic'], semanticTokens) 
        with tab2:
                render_tab(query, "Keyword", sl.session_state.respostas['Keyword'], keywordTokens)
        with tab3:
                render_tab(query, "Hybrid", sl.session_state.respostas['Hybrid'], hybridTokens)
                