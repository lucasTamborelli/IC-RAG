from classes import *



if __name__ == "__main__":
    
    arquivo = 'doc_ele01.pdf'
    llm = LLM_cloud('gpt-oss:120b-cloud', temperature = 0.1)

    documento = Treater(arquivo)
    text = documento.extract_text()
    chunks = documento.split_chunks(text, chunk_size = 5000, overlap = 1000)
   
    query = 'Quantas disciplinas são cursadas no 1o Ano Profissional e qual a carga horária semanal teórica'


    semantic_context = semantic_search(index_name='teste', text_chunks=chunks, query=query, top_k = 5, printer= False)
    # 1000, 200

    keyword_context = keyword_search(index_name='teste22', text_chunks=chunks, query=query, top_k = 5, printer= False)
    # 1000, 400 # overlap maior


    hybrid_context = hybrid_search(keyword_context, semantic_context, sparse_weight = 0.6, top_k = 3, printer = False)


    answer = llm.response(query, hybrid_context)
    print("\n\n=============================\n\n")
    print(answer)
    print("\n\n=============================\n\n")