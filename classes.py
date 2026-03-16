import os
from langchain_ollama import ChatOllama
from pypdf import PdfReader
from ollama import Client
import streamlit as sl
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["USER_AGENT"] = "rag"

if "GEMINI_API_KEY" in sl.secrets:
    gemini_key = sl.secrets["GEMINI_API_KEY"]
else:
    gemini_key = os.getenv("GEMINI_API_KEY")

class LLM_cloud():
    def __init__(self, model, temperature):
        self.model = model
        self.temp = temperature
        self.model = ChatGoogleGenerativeAI(
            model=model, 
            temperature=temperature, 
            google_api_key=gemini_key
        )
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
