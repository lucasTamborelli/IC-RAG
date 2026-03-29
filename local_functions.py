from dotenv import load_dotenv
import tiktoken
from typing import *
import pickle
import os
import streamlit as sl
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# @sl.cache_resource
def load_databases():
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        faiss_db = FAISS.load_local("vector_db/faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        with open("vector_db/bm25_retriever.pkl", "rb") as f:
                bm25_retriever = pickle.load(f)
                
        return faiss_db, bm25_retriever

def semantic_search(faiss_db, query, top_k=3):
        results = faiss_db.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

def keyword_search(bm25_retriever, query, top_k=3):
        bm25_retriever.k = top_k
        results = bm25_retriever.invoke(query)
        return [doc.page_content for doc in results]

def hybrid_search(faiss_db, bm25_retriever, query, top_k=3):
        sem_results = semantic_search(faiss_db, query, top_k=top_k)
        key_results = keyword_search(bm25_retriever, query, top_k=top_k)
        
        hybrid_context = []
        for text in sem_results + key_results:
                if text not in hybrid_context:
                        hybrid_context.append(text)
                        
        return hybrid_context[:top_k]

def render_tab(type, answer, tokens):
	sl.markdown(f"**Resposta:**")
	sl.write(answer)
	sl.text(f"Tokens: {tokens}")

def n_tokens(input: str, output: str, model_name: str):
	enc = tiktoken.get_encoding(model_name)
	tokens_in = enc.encode(input)
	tokens_out = enc.encode(output)
	return [len(tokens_in), len(tokens_out)]

