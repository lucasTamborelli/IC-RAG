import os
import sys
import glob
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.retrieval.preprocess import preprocess_pt
from src.ingestion.doc_loader import Treater

"""
Roda-se uma unica vez esse .py para popular a base de dados
"""

load_dotenv()

def extract_and_chunk_documents(directory: str):
        document_paths = glob.glob(os.path.join(directory, '*.pdf'))

        all_chunks = []
        for path in document_paths:
                treater = Treater(path)
                chunks = treater.split_chunks(chunk_size=1000, overlap=150) # era 5000 com 1000
                print(f"  → {len(chunks)} chunks de '{os.path.basename(path)}'")
                all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":

        directory = './data/raw_IPR'
        chunks = extract_and_chunk_documents(directory)

        if not chunks:
                exit()
        print(f"\nTotal: {len(chunks)} chunks. Populando bases...")
        os.makedirs("data/vector_db/faiss_index", exist_ok=True)

        # Semantic Index
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        faiss_db = FAISS.from_documents(chunks, embeddings)
        faiss_db.save_local("data/vector_db/faiss_index")
        print("FAISS salvo")

        # Keyword Index
        bm25_retriever = BM25Retriever.from_documents(
                chunks,
                preprocess_func=preprocess_pt
        )
        with open("data/vector_db/bm25_retriever.pkl", "wb") as f:
                pickle.dump(bm25_retriever, f)

        print("BM25 salvo")

        print("\nBase de dados criada com sucesso.")