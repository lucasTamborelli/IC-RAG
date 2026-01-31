import os
os.environ["USER_AGENT"] = "rag"

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


from pypdf import PdfReader
from pinecone import Pinecone
from collections import defaultdict
from ollama import Client


from dotenv import load_dotenv

load_dotenv()
keyPinecone = os.getenv("API_KEY_PINECONE")


class LLM_cloud():
    def __init__(self, model, temperature):
        self.model = model
        self.temp = temperature
        self.client = Client()
        # model = 'gpt-oss:120b-cloud'

    def prompt(self, query, context):
        prompt = f"""
        Utilize o contexto para responder à pergunta abaixo.

        Pergunta:
        {query}

        Contexto:
        {context}
        """
        return prompt.strip()

    def response(self, query, context):
        rag_prompt = self.prompt(query, context)
        messages = [
            {
                "role": "user",
                "content": rag_prompt,
            }
        ]
        response = self.client.chat(self.model, messages=messages)
        return response["message"]["content"]



class LLM():
    def __init__(self, model, temperature):
        self.temp = temperature
        self.model = ChatOllama(model = model, temperature = temperature) 
        # llama3.2:1b  llama3.1

    def prompt(self, query, context):

        prompt = f'''
        Utilize o contexto para responder a sobre a
        Pergunta:
        {query}

        Contexto:
        {context}

        '''
        return prompt
    
    def response(self, query, context):
        rag_prompt = self.prompt(query, context)
        response = self.model.invoke(rag_prompt) 
        return response.content


class Treater():
    def __init__(self, file):
        self.file = file

    def extract_text(self):
        return "".join([p.extract_text() for p in PdfReader(self.file).pages])
    
    # json estruturado1

    def split_chunks(self, text, chunk_size, overlap): # chunk_size=1000, overlap=200
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap 
        return chunks




def semantic_search(index_name, text_chunks, query, top_k, printer):

    pc = Pinecone(api_key= keyPinecone)
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud = 'aws',
            region = 'us-east-1',
            embed = {
                'model':'llama-text-embed-v2',
                'field_map':{'text': 'chunk_text'}
            }
        )
    records = [
        {
            "_id": f"chunk_{i+1}",
            "chunk_text": chunk,
            "category": "pdf_content"
        }
        for i, chunk in enumerate(text_chunks)
    ]
    dense_index = pc.Index(index_name)
    dense_index.upsert_records('teste1', records)
    results = dense_index.search(
        namespace="teste1",
        query={
            "top_k": top_k,
            "inputs": {
                'text': query
            }
        }
    )
    context_texts = []
    for hit in results['result']['hits']:
            if printer:
                print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")     
            context_texts.append(hit['fields']['chunk_text'])

    # dense_context = "\n\n".join(context_texts)
    # return dense_context
    return results  # <-- Return the full results dict


def keyword_search(index_name, text_chunks, query, top_k, printer):
    
    pc = Pinecone(api_key= keyPinecone)
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            # metric = "dotproduct",
            embed={
                "model": "pinecone-sparse-english-v0",
                "field_map": {"text": "chunk_text"}
            }
        )
    records = [
        {
            "_id": f"chunk_{i+1}",
            "chunk_text": chunk,
            "category": "pdf_content"
        }
        for i, chunk in enumerate(text_chunks)
    ]
    sparse_index = pc.Index(index_name)
    sparse_index.upsert_records(namespace="pdf", records=records)
    results = sparse_index.search(
        namespace="pdf",
        query={
            "top_k": top_k,
            "inputs": {"text": query}
        }
    )
    context_texts = []
    for hit in results["result"]["hits"]:
        if printer:
            print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")    
    
    # sparse_context = "\n\n".join(context_texts)
    # return sparse_context
    return results  



def hybrid_search(dense_result, sparse_result, sparse_weight, top_k, printer):
    final_score = defaultdict(lambda: {"text": "", "dense": 0, "sparse": 0}) 

    for hit in dense_result["result"]["hits"]:
        _id = hit["_id"]
        final_score[_id]["text"] = hit["fields"]["chunk_text"]
        final_score[_id]["dense"] = hit["_score"]    


    for hit in sparse_result["result"]["hits"]:
        _id = hit["_id"]
        final_score[_id]["text"] = hit["fields"]["chunk_text"]
        final_score[_id]["sparse"] = hit["_score"]


    final = []
    for _id, vals in final_score.items():
        combined_score = sparse_weight * vals["sparse"] + (1 - sparse_weight) * vals["dense"]
        final.append({"_id": _id, "text": vals["text"], "score": combined_score})

    final = sorted(final, key=lambda x: x["score"], reverse=True)
    top_k_final = final[:top_k]
    
    for hit in top_k_final:
        if printer: 
            print(f"id: {hit['_id']} | score: {hit['score']:.2f} | text: {hit['text'][:500]}...")
    return top_k_final

