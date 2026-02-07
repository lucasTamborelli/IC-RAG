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




    if 'resposta_llm' not in sl.session_state: sl.session_state.resposta_llm = None
    if 'ultima_query' not in sl.session_state: sl.session_state.ultima_query = ""

    query = sl.text_input(label='Pergunta:')
    buscar = sl.button('Buscar') 
    if buscar or (query and query != sl.session_state.ultima_query):
        semanticCtx = semantic_search(index_name='teste', text_chunks=chunks, query=query, top_k = 5, printer= False) # 1000, 200
        keywordCtx = keyword_search(index_name='teste22', text_chunks=chunks, query=query, top_k = 5, printer= False) # 1000, 400
        # sparce == semantic -> 60% semantic
        hybridCtx = hybrid_search(keywordCtx, semanticCtx, sparse_weight = 0.6, top_k = 3, printer = False)
        answer = llm.response(query, hybridCtx)
        sl.write(answer)

        sl.session_state.resposta_llm = answer
        sl.session_state.ultima_query = query


        if sl.session_state.resposta_llm:
            sl.write(sl.session_state.resposta_llm)
            sl.write("---") 

            evaluation = sl.radio( 'Avaliação da resposta:', ("Ruim", "Médio", "Bom"),
                key='avaliacao_widget',
                on_change=save_feedback,
                index=None,
                args=('feedback.csv', query, answer),
            )
            