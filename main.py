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

    query = 'Qual ação o coordenador deve tomar se a proposta técnica NÃO for aprovada pela empresa?'


    semantic_context = semantic_search(index_name='teste', text_chunks=chunks, query=query, top_k = 5, printer= False)
    # 1000, 200

    keyword_context = keyword_search(index_name='teste22', text_chunks=chunks, query=query, top_k = 5, printer= False)
    # 1000, 400 # overlap maior


    hybrid_context = hybrid_search(keyword_context, semantic_context, sparse_weight = 0.6, top_k = 3, printer = False)


    answer = llm.response(query, hybrid_context)
    print("\n\n=============================\n\n")
    print(answer)
    print("\n\n=============================\n\n")