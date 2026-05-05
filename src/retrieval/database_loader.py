import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_databases():
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        faiss_db = FAISS.load_local(
                "data/vector_db/faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
        )
        with open("data/vector_db/bm25_retriever.pkl", "rb") as f:
                bm25_retriever = pickle.load(f)
        return faiss_db, bm25_retriever
