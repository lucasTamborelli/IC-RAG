from classes import *

if __name__ == "__main__":
    
    directory = './IPRdocuments'
    documentList = glob.glob(os.path.join(directory, '*.pdf'))

    for archive in documentList:
        document = Treater(archive)
        text = document.extract_text()
        if text:
            chunks = document.split_chunks(text, chunk_size = 5000, overlap = 1000)

    llm = LLM_cloud('gpt-oss:120b-cloud', temperature = 0.1)

    sl.header("Chat infos IPR")
    query = sl.text_input(label='Pergunta:')
    if query: 
        semantic_context = semantic_search(index_name='teste', text_chunks=chunks, query=query, top_k = 5, printer= False) # 1000, 200
        keyword_context = keyword_search(index_name='teste22', text_chunks=chunks, query=query, top_k = 5, printer= False) # 1000, 400
        # sparce == samantic -> 60% semantic
        hybrid_context = hybrid_search(keyword_context, semantic_context, sparse_weight = 0.6, top_k = 3, printer = False)
        answer = llm.response(query, hybrid_context)
        sl.write(answer)
