from classes import *
from local_functions import *
from db_populator import *

### pergunta base de teste: Qual é a finalidade principal da norma NPA-ITA-070:2024?
# Qual é a finalidade principal da norma NPA ITA - 070 

if __name__ == "__main__":
        
        llm = LLM_cloud(model="gpt-4.1", temperature = 0.2) # gpt-4o

        faiss_db, bm25_retriever = load_databases()


        while True:
                query = input('Pergunte sobre os documentos da IPR\n\n')
                

                sem_chunks = semantic_search(faiss_db, query, top_k=3)
                key_chunks = keyword_search(bm25_retriever, query, top_k=3)
                hyb_chunks = hybrid_search(faiss_db, bm25_retriever, query, top_k=3)
                
                ctx_semantico = "\n\n".join(sem_chunks)
                ctx_keyword = "\n\n".join(key_chunks)
                ctx_hibrido = "\n\n".join(hyb_chunks)
                
                semantic_response = llm.response(query, ctx_semantico)
                semanticTokens = n_tokens(llm.prompt(query, ctx_semantico), semantic_response, "o200k_base")
                
                keyword_response = llm.response(query, ctx_keyword)
                keywordTokens = n_tokens(llm.prompt(query, ctx_keyword), keyword_response, "o200k_base")
                
                hybrid_response = llm.response(query, ctx_hibrido)
                hybridTokens = n_tokens(llm.prompt(query, ctx_hibrido), hybrid_response, "o200k_base")
                

                print("semantic\n", semantic_response, semanticTokens)
                print("keyword\n", keyword_response, keywordTokens)
                print("hybrid\n", hybrid_response, hybridTokens)