import os
import sys
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DEFAULT_EMBEDDING, VECTOR_DB_DIR


def load_databases(
        db_directory: str = VECTOR_DB_DIR,
        embedding_key: str = DEFAULT_EMBEDDING,
):
        """
        Carrega FAISS + BM25 de um diretório.
        embedding_key deve corresponder ao modelo da criação do índice.
        """
        from src.ingestion.db_populator import get_embeddings

        embeddings = get_embeddings(embedding_key)
        faiss_db = FAISS.load_local(
                os.path.join(db_directory, "faiss_index"),
                embeddings,
                allow_dangerous_deserialization=True,
        )
        bm25_path = os.path.join(db_directory, "bm25_retriever.pkl")
        with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
        return faiss_db, bm25_retriever
