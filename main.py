import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import BaseModel, Field
from typing import Dict
import uuid

# --- Configuración de Streamlit ---
st.set_page_config(page_title="💊 COFAS", layout="centered")
st.title("💊 COFAS - Asistente Farmacéutico Inteligente")

# --- Entrada de credenciales ---
st.subheader("🔐 Introduce tus credenciales")
openai_key = st.text_input("🔑 OpenAI API Key", type="password", key="openai_key_input")
elastic_key = st.text_input("🔑 Elasticsearch API Key", type="password", key="elastic_key_input")

ELASTIC_ENDPOINT = "https://a651d368f9a74262abe48b46287674d6.europe-west3.gcp.cloud.es.io:443"
INDEX = "cima-index"

# --- Función para construir query Elasticsearch ---
def createQuery(question: str):
    return {
        "query": {
            "multi_match": {
                "query": question,
                "fields": ["text"]
            }
        },
        "size": 5,
        "_source": ["text", "nombre", "prescripcion", "dosis", "receta", "fotos"]
    }

# --- Función de búsqueda en Elasticsearch ---
def elastic_search(question: str):
    global es_client
    if es_client is None:
        return "El cliente Elasticsearch no está inicializado."
    query = createQuery(question)
    response = es_client.search(index=INDEX, body=query)
    hits = response["hits"]["hits"]
    if not hits:
        return "No se encontraron resultados para tu pregunta."
    result_docs = []
    for hit in hits:
        source = hit["_source"]
        name = source.get("nombre", "")
        text = source.get("text", "")
        doc = f"Según la base de datos, el medicamento '{name}' tiene la siguiente información relevante: {text}"
        result_docs.append(doc)
    return f"Usa esta información para responder a la pregunta: {question} \n" + "\n".join(result_docs)

# --- Tool para LangChain ---
class RagSearchInput(BaseModel):
    question: str = Field(..., description="Consulta de búsqueda sobre medicamentos o dolencias.")

rag_search_tool = StructuredTool(
    name="RAG_Search",
    func=elastic_search,
    description="Busca información en la base de datos sobre un medicamento o dolencia.",
    args_schema=RagSearchInput
)

# --- Prompt del agente ---
context = """
Eres un asistente de una farmacia.
Responde a la pregunta: {input}

Este es el historial de conversación reciente:
{history}

Responde siempre en un párrafo de manera amigable y profesional.

Puedes acceder a información adicional si lo necesitas:
{agent_scratchpad}
"""

# --- Inicialización de servicios si hay claves ---
if openai_key and elastic_key:
    try:
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-4o-mini",  # Puedes ajustar a otro modelo disponible
            temperature=0.1,
        )

        es_client = Elasticsearch(
            ELASTIC_ENDPOINT,
            api_key=elastic_key
        )

        tools = [rag_search_tool]

        prompt = PromptTemplate(
            input_variables=["input", "history", "agent_scratchpad"],
            template=context
        )

        if "agent" not in st.session_state:
            st.session_state.agent = create_openai_tools_agent(llm, tools, prompt)

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(k=3, return_messages=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.success("✅ Conectado correctamente a OpenAI y Elasticsearch.")

    except Exception as e:
        st.error(f"❌ Error al conectar: {e}")

else:
    st.warning("🔒 Introduce ambas claves para iniciar el asistente.")

# --- Interfaz de conversación ---
if openai_key and elastic_key and "agent" in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("¿En qué puedo ayudarte hoy?")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        agent_executor = AgentExecutor(
            agent=st.session_state.agent,
            tools=tools,
            memory=st.session_state.memory,
            verbose=False,
            streaming=True,
            max_iterations=5

        )

        with st.chat_message("assistant"):
            response_container = st.empty()
            final_response = ""

            response = agent_executor.invoke({"input": user_input})

            final_response = response["output"]
            if final_response == "Agent stopped due to max iterations.":
                final_response = "Actualmente no tengo información en mi sistema como para poder responder a esa consulta."
            
            response_container.markdown(final_response)

        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
