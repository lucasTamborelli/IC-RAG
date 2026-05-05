import os
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
                Você é um assistente especializado em documentos de Propriedade Intelectual do ITA.
                Utilize APENAS o contexto fornecido para responder à pergunta. 
                Se a resposta não estiver no contexto, diga que não encontrou a informação.
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