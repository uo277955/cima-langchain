from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import BaseModel, Field
from typing import Dict
import uuid
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


# Entrada de claves API
st.subheader("🔐 Introduce tus credenciales")

openai_key = st.text_input("🔑 OpenAI API Key", type="password", key="openai_key_input")
elastic_key = st.text_input("🔑 Elasticsearch API Key", type="password", key="elastic_key_input")

ELASTIC_ENDPOINT="https://a651d368f9a74262abe48b46287674d6.europe-west3.gcp.cloud.es.io:443"
INDEX = "cima-index"

#Creamos el modelo llm
llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini-2024-07-18", temperature = 0.5)

#Acceso a elasticsearch
es_client = Elasticsearch(
    ELASTIC_ENDPOINT,
    api_key=elastic_key
)


def createQuery(question: str):
    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "multi_match": {
                        "query": question,
                        "fields": [
                            "text"
                        ]
                    }
                }
            }
        },
        "size": 3,
        "_source": ["text", "nombre", "prescripcion", "dosis", "receta", "fotos"]
    }
    
    return es_query

def elastic_search(question:str):
    print(question)
    if es_client is None:
        return "El cliente no esta inicializado."
    else:
        #Construimos la query de elastic
        query = createQuery(question)
        response = es_client.search(index=INDEX, body=query)
        hits = response["hits"]["hits"]
        if not hits:
            return "No se encuentran respuestas para la query."
        result_docs = []
        for hit in hits:
            source = hit["_source"]
            name = source.get("nombre", "")
            text = source.get("text", "")
            doc = f"Según la base de datos, el medicamento '{name}' tiene la siguiente información relevante: {text}"
            result_docs.append(doc)
        return f"Usa esta información para responder a la pregunta: {question} \n" + "\n".join(result_docs)



class RagSearchInput(BaseModel):
    question:str = Field(..., description="La query de busqueda para obtener el conocimiento")
rag_search_tool = StructuredTool(
    name="RAG_Search",
    func=elastic_search,
    description=(
        "Usa esta función para buscar información sobre un medicamento o una dolencia. "
        "**Input debe de contener la pergunta o la cuestion que hace el usuario enriquecida con información adicional extraida de la conversación si es necesario** "
    ),
    args_schema=RagSearchInput
)

# List of tools
tools = [rag_search_tool]

context = """
        Eres un asistente de una farmacia.
        Reponde a la pregunta {input}
        
        Este es el historial de conversación reciente:
        {history}
        
        Responde siempre en un párrafo de manera amigable
        
        Tienes acceso al historial
        {agent_scratchpad}
"""

memory = ConversationBufferWindowMemory(k=3, return_messages=True)


prompt = PromptTemplate(
    input_variables=["input", "history", "agent_scratchpad"],
    template=context
)

agent = create_openai_tools_agent(llm, tools, prompt)

# Diccionario para guardar la memoria por sesión
sessions_memory: Dict[str, ConversationBufferWindowMemory] = {}


# --- Streamlit UI ---
st.set_page_config(page_title="COFAS", layout="centered")
st.title("💊 COFAS")

# Inicializar sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar mensajes anteriores
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
user_prompt = st.chat_input("¿Qué necesitas saber hoy?")
if user_prompt:
    # Mostrar mensaje del usuario de inmediato
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Preparar y ejecutar agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=st.session_state.memory,
        verbose=False,
        streaming=True,
    )

    with st.chat_message("assistant"):
        response_container = st.empty()
        final_response = ""

        # Captura de respuesta en tiempo real
        for chunk in agent_executor.stream({"input": user_prompt}):
            token = chunk.get("output", "")
            final_response += token
            response_container.markdown(final_response + "▌")

        response_container.markdown(final_response)

    # Guardar respuesta final
    st.session_state.chat_history.append({"role": "assistant", "content": final_response})