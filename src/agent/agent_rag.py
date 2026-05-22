"""
Agentic RAG com LangChain 1.x create_agent
===========================================

Comparação com a imagem (CrewAI + Qwen3):
  Imagem:    Retriever Agent → tools (Qdrant, Firecrawl) → Notes → Response Generator Agent
  Este arquivo: Um único agente LangChain com 3 ferramentas de busca nos documentos.

A diferença chave para o pipeline atual (main.py):
  Atual:    query → executa SEMPRE as 3 buscas → LLM responde com contexto fixo
  Agente:   query → LLM DECIDE quais ferramentas chamar e quando → LLM responde

LangChain 1.x usa create_agent (substitui create_tool_calling_agent + AgentExecutor).
O agente retorna um CompiledStateGraph que é invocado via .invoke({"messages": [...]}).
"""

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from retrieval import load_databases, semantic_search, keyword_search, hybrid_search

load_dotenv()

# ──────────────────────────────────────────────────────────
# Carrega bases (apenas uma vez)
# ──────────────────────────────────────────────────────────
faiss_db, bm25_retriever = load_databases()


# ──────────────────────────────────────────────────────────
# Ferramentas disponíveis ao agente
# Cada @tool vira um "tool call" que o LLM pode invocar.
# A docstring é o que o modelo lê para decidir quando usar.
# ──────────────────────────────────────────────────────────

@tool
def busca_semantica(query: str) -> str:
    """
    Busca documentos por similaridade semântica usando embeddings vetoriais.
    Use para perguntas conceituais, definições, finalidades, e quando a
    pergunta não contém termos técnicos exatos que aparecem nos documentos.
    Exemplos: 'qual a finalidade da norma', 'o que é propriedade intelectual'
    """
    chunks = semantic_search(faiss_db, query, top_k=4)
    if not chunks:
        return "Nenhum resultado encontrado na busca semântica."
    resultado = "\n\n---\n\n".join(chunks)
    return f"[Busca Semântica - {len(chunks)} trechos encontrados]\n\n{resultado}"


@tool
def busca_palavras_chave(query: str) -> str:
    """
    Busca documentos por palavras-chave exatas usando BM25.
    Use para termos técnicos específicos, siglas, números de artigos/normas,
    nomes próprios ou quando você precisa de trechos que contenham palavras exatas.
    Exemplos: 'NPA-ITA-070', 'Art. 5', 'cessão de direitos', 'royalties'
    """
    chunks = keyword_search(bm25_retriever, query, top_k=4)
    if not chunks:
        return "Nenhum resultado encontrado na busca por palavras-chave."
    resultado = "\n\n---\n\n".join(chunks)
    return f"[Busca por Palavras-chave - {len(chunks)} trechos encontrados]\n\n{resultado}"


@tool
def busca_hibrida(query: str) -> str:
    """
    Combina busca semântica e por palavras-chave com score ponderado.
    Use quando a pergunta mistura conceitos e termos específicos, ou quando
    as buscas individuais não retornaram resultados satisfatórios.
    """
    chunks = hybrid_search(faiss_db, bm25_retriever, query, sparse_weight=0.5, top_k=4)
    if not chunks:
        return "Nenhum resultado encontrado na busca híbrida."
    resultado = "\n\n---\n\n".join(chunks)
    return f"[Busca Híbrida - {len(chunks)} trechos encontrados]\n\n{resultado}"


# ──────────────────────────────────────────────────────────
# Criação do agente (API LangChain 1.x)
# create_agent substitui create_tool_calling_agent + AgentExecutor.
# Retorna um CompiledStateGraph; invocado via .invoke({"messages": [...]}).
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um assistente especializado em documentos de Propriedade Intelectual do ITA.

Você tem acesso a ferramentas de busca nos documentos oficiais da IPR.

Instruções:
1. Sempre use pelo menos uma ferramenta de busca antes de responder.
2. Escolha a ferramenta mais adequada para o tipo de pergunta.
3. Se o resultado não for satisfatório, tente outra ferramenta ou reformule a query.
4. Responda APENAS com base nas informações encontradas nos documentos.
5. Se não encontrar a informação, diga claramente que não está disponível."""

agent = create_agent(
    model="openai:gpt-4o",
    tools=[busca_semantica, busca_palavras_chave, busca_hibrida],
    system_prompt=SYSTEM_PROMPT,
)


def visualizar_grafo():
    """
    Mostra o grafo do agente e detalhes das ferramentas disponíveis.
    """
    print("\n" + "="*60)
    print("ESTRUTURA DO AGENTE AGENTIC RAG")
    print("="*60 + "\n")
    
    print("📊 Grafo LangGraph (ASCII):")
    print("-" * 60)
    try:
        agent.get_graph().print_ascii()
    except Exception as e:
        print(f"Não foi possível gerar ASCII: {e}")
    print()
    
    print("🔧 Ferramentas disponíveis ao agente:")
    print("-" * 60)
    tools = [busca_semantica, busca_palavras_chave, busca_hibrida]
    for i, tool_func in enumerate(tools, 1):
        print(f"\n{i}. {tool_func.name}")
        print(f"   Descrição: {tool_func.description[:200]}...")
    
    print("\n" + "="*60)
    print("FLUXO DETALHADO (Mermaid):")
    print("="*60)
    print("""
graph TD
    START([Usuário faz pergunta]) --> MODEL[🤖 GPT-4o lê pergunta e system prompt]
    
    MODEL --> DECIDE{Decisão do modelo}
    
    DECIDE -->|"Pergunta conceitual"| T1[🔍 busca_semantica]
    DECIDE -->|"Termos técnicos/siglas"| T2[🔍 busca_palavras_chave]
    DECIDE -->|"Query complexa"| T3[🔍 busca_hibrida]
    
    T1 --> FAISS[(FAISS Vector DB<br/>text-embedding-3-small)]
    T2 --> BM25[(BM25 Index<br/>preprocessamento PT)]
    T3 --> HYBRID[Combina FAISS + BM25<br/>weighted score]
    
    FAISS --> CHUNKS1[Retorna top-4 chunks]
    BM25 --> CHUNKS2[Retorna top-4 chunks]
    HYBRID --> CHUNKS3[Retorna top-4 chunks]
    
    CHUNKS1 --> MODEL_RESPONSE[🤖 GPT-4o gera resposta<br/>baseada nos chunks]
    CHUNKS2 --> MODEL_RESPONSE
    CHUNKS3 --> MODEL_RESPONSE
    
    MODEL_RESPONSE --> CHECK{Resultado satisfatório?}
    CHECK -->|Sim| END([Retorna resposta final])
    CHECK -->|Não| DECIDE
    
    style START fill:#e1f5e1
    style END fill:#e1f5e1
    style MODEL fill:#fff3cd
    style MODEL_RESPONSE fill:#fff3cd
    style T1 fill:#cfe2ff
    style T2 fill:#cfe2ff
    style T3 fill:#cfe2ff
    style FAISS fill:#f8d7da
    style BM25 fill:#f8d7da
    style HYBRID fill:#f8d7da
""")
    print("\nCopie o código acima e cole em https://mermaid.live para visualizar")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Agente RAG - IPR ITA")
    print("Digite 'sair' para encerrar")
    print("Digite 'grafo' para visualizar a estrutura detalhada do agente\n")

    while True:
        query = input("Pergunta: ").strip()
        
        if query.lower() in ("sair", "exit", "quit"):
            break
        
        if query.lower() == "grafo":
            visualizar_grafo()
            continue
            
        if not query:
            continue

        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # A resposta final está na última mensagem do histórico
        last_message = result["messages"][-1]
        print("\n" + "="*60)
        print("RESPOSTA FINAL:")
        print(last_message.content)
        print("="*60 + "\n")
        