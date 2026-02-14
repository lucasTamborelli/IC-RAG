from classes import *
from functions import *
import streamlit as sl
import glob



if __name__ == "__main__":
    
    directory = './IPRdocuments'
    documentList = glob.glob(os.path.join(directory, '*.pdf'))

    for archive in documentList:
        document = Treater(archive)
        text = document.extract_text()
        if text:
            chunks = document.split_chunks(text, chunk_size = 5000, overlap = 1000)
            
    llm = LLM_cloud('gpt-oss:120b-cloud', temperature = 0.1)


    if 'resposta_llm' not in sl.session_state: 
        sl.session_state.respostas = {"Semantic": None, "Keyword": None, "Hybrid": None}
    if 'ultima_query' not in sl.session_state:
        sl.session_state.ultima_query = ""

    query = sl.text_input(label='Pergunta:')
    buscar = sl.button('Buscar') 
    if buscar or (query and query != sl.session_state.ultima_query):
        semanticCtx = semantic_search(index_name='teste', text_chunks=chunks, query=query, top_k = 5, printer= False)
        sl.session_state.respostas['Semantic'] = llm.response(query, semanticCtx)

        keywordCtx = keyword_search(index_name='teste22', text_chunks=chunks, query=query, top_k = 5, printer= False)
        sl.session_state.respostas['Keyword'] = llm.response(query, keywordCtx)

        hybridCtx = hybrid_search(keywordCtx, semanticCtx, sparse_weight = 0.6, top_k = 3, printer = False)
        sl.session_state.respostas['Hybrid'] = hybridAnswer = llm.response(query, hybridCtx)

        sl.session_state.ultima_query = query

        if sl.session_state.respostas['Semantic']:
            tab1, tab2, tab3 = sl.tabs(['Semantic', 'Keyword', 'Hybrid'])

            render_tab(query, tab1, "Semantic", sl.session_state.respostas['Semantic'])
            render_tab(query, tab2, "Keyword", sl.session_state.respostas['Keyword'])
            render_tab(query, tab3, "Hybrid", sl.session_state.respostas['Hybrid'])
            