import os
import sys
import glob
import pickle
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from src.retrieval.preprocess import preprocess_pt
from src.ingestion.doc_loader import Treater
from src.config import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_EMBEDDING,
        DEFAULT_OVERLAP,
        EMBEDDING_CONFIGS,
        RAW_PDF_DIR,
        VECTOR_DB_DIR,
)

load_dotenv()


def get_embeddings(embedding_key: str = DEFAULT_EMBEDDING):
        """Retorna o objeto de embeddings para o provider/modelo especificado."""
        config = EMBEDDING_CONFIGS[embedding_key]

        if config["provider"] == "openai":
                return OpenAIEmbeddings(model=config["model"])

        elif config["provider"] == "voyage":
                from langchain_voyageai import VoyageAIEmbeddings
                return VoyageAIEmbeddings(model=config["model"])

        raise ValueError(f"Provider desconhecido: {config['provider']}")


def extract_and_chunk_documents(
        directory: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
):
        document_paths = glob.glob(os.path.join(directory, '*.pdf'))

        all_chunks = []
        for path in document_paths:
                treater = Treater(path)
                chunks = treater.split_chunks(chunk_size=chunk_size, overlap=overlap)
                print(f"  -> {len(chunks)} chunks de '{os.path.basename(path)}'")
                all_chunks.extend(chunks)
        return all_chunks


def populate_databases(
        pdf_directory: str = RAW_PDF_DIR,
        output_directory: str = VECTOR_DB_DIR,
        embedding_key: str = DEFAULT_EMBEDDING,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
):
        """
        Popula FAISS + BM25
        Retorna (faiss_db, bm25_retriever) para uso direto
        """
        chunks = extract_and_chunk_documents(pdf_directory, chunk_size, overlap)

        if not chunks:
                print("Nenhum chunk extraído. Abortando.")
                return None, None

        print(f"\nTotal: {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

        faiss_dir = os.path.join(output_directory, "faiss_index")
        os.makedirs(faiss_dir, exist_ok=True)

        print(f"Embedding: {embedding_key} ({EMBEDDING_CONFIGS[embedding_key]['model']})")
        embeddings = get_embeddings(embedding_key)
        faiss_db = FAISS.from_documents(chunks, embeddings)
        faiss_db.save_local(faiss_dir)
        print("FAISS salvo")

        bm25_retriever = BM25Retriever.from_documents(
                chunks,
                preprocess_func=preprocess_pt,
        )
        bm25_path = os.path.join(output_directory, "bm25_retriever.pkl")
        with open(bm25_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
        print("BM25 salvo")

        print("\nBase de dados criada")
        return faiss_db, bm25_retriever


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Popula indices FAISS + BM25")
        parser.add_argument("--embedding", default=DEFAULT_EMBEDDING,
                            choices=list(EMBEDDING_CONFIGS.keys()))
        parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
        parser.add_argument("--output-dir", default=VECTOR_DB_DIR)
        args = parser.parse_args()

        populate_databases(
                embedding_key=args.embedding,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                output_directory=args.output_dir,
        )