import os
from langchain_ollama import ChatOllama
from pypdf import PdfReader
from ollama import Client


os.environ["USER_AGENT"] = "rag"

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
