import os
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


os.environ["USER_AGENT"] = "rag"
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


class LLM_cloud():
    def __init__(self, model, temperature):
        self.model = model
        self.temp = temperature
        self.model = ChatOpenAI(
        	model = model, 
        	temperature = temperature, 
        	openai_api_key = openai_key
        )

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
        response = self.model.invoke(messages)
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
