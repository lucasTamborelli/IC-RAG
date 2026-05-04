import os
import glob
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from classes import Treater


"""
# Roda-se uma unica vez esse .py para popular a base de dados
"""

load_dotenv()

def extract_and_chunk_documents(directory: str):
        documentList = glob.glob(os.path.join(directory, '*.pdf'))
        
        all_chunks = []
        for archive in documentList:
                document = Treater(archive)
                text = document.extract_text()
                if text:
                        chunks = document.split_chunks(text, chunk_size=5000, overlap=1000)
                        print(f"  → {len(chunks)} chunks extraídos de '{os.path.basename(archive)}'")
                        all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
        
        directory = './IPRdocuments'
        chunks = extract_and_chunk_documents(directory)
        if not chunks: exit()
        os.makedirs("vector_db", exist_ok=True)

        # Semantic Index
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        faiss_db = FAISS.from_texts(chunks, embeddings)
        faiss_db.save_local("vector_db/faiss_index")
        
        # Keyword Index
        bm25_retriever = BM25Retriever.from_texts(chunks)
        with open("vector_db/bm25_retriever.pkl", "wb") as f:
                pickle.dump(bm25_retriever, f)
                
        print("Created")