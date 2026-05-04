from collections import defaultdict
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

def hybrid_search(faiss_db, bm25_retriever, query, sparse_weight=0.5, top_k=3):
        sem_scored = faiss_db.similarity_search_with_score(query, k=top_k)
        bm25_retriever.k = top_k
        key_results = bm25_retriever.invoke(query)

        final_scores = defaultdict(lambda: {"text": "", "dense": 0.0, "sparse": 0.0})

        for doc, distance in sem_scored:
                text = doc.page_content
                final_scores[text]["text"] = text
                final_scores[text]["dense"] = 1 / (1 + distance)

        for rank, doc in enumerate(key_results):
                text = doc.page_content
                final_scores[text]["text"] = text
                final_scores[text]["sparse"] = (top_k - rank) / top_k

        final = []
        for text, vals in final_scores.items():
                combined = sparse_weight * vals["sparse"] + (1 - sparse_weight) * vals["dense"]
                final.append({"text": vals["text"], "score": combined})

        final.sort(key=lambda x: x["score"], reverse=True)
        return [item["text"] for item in final[:top_k]]

def render_tab(type, answer, tokens):
	sl.markdown(f"**Resposta:**")
	sl.write(answer)
	sl.text(f"Tokens: {tokens}")

def n_tokens(input: str, output: str, model_name: str):
	enc = tiktoken.get_encoding(model_name)
	tokens_in = enc.encode(input)
	tokens_out = enc.encode(output)
	return [len(tokens_in), len(tokens_out)]

