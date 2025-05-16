import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
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
    print(question)
    return {
        "query": {
            "multi_match": {
                "query": question,
                "fields": ["text"]
            }
        },
        "size": 3,
        "_source": ["text", "nombre", "prescripcion", "dosis", "receta", "fotos", "documento"]
    }

# --- Función de búsqueda en Elasticsearch ---
def elastic_search(question: str):
    global es_client
    st.session_state.recommendations = []
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
        print(source)
        st.session_state.recommendations.append(source.get("documento"))
    return f"Usa esta información para responder a la pregunta: {question} \n" + "\n".join(result_docs)

# --- Tool para LangChain ---

class RagSearchInput(BaseModel):
    question:str = Field(..., description="La query de busqueda para obtener el conocimiento")
rag_search_tool = StructuredTool(
    name="RAG_Search",
    func=elastic_search,
    description=(
        "Usa esta función para buscar información sobre un medicamento o una dolencia. "
        "**Input debe de contener la pregunta o la cuestion que hace el usuario si es necesario enriquecida con un poco de contexto** "
    ),
    args_schema=RagSearchInput
)

# --- Prompt del agente ---
context = context = """
Eres un asistente experto de farmacia, cordial, claro y útil.

Tu función principal es ayudar a los usuarios a encontrar información sobre medicamentos, dolencias y tratamientos, basándote en el conocimiento disponible o haciendo búsquedas si es necesario.

Responde a la pregunta del usuario: {input}

Historial reciente de la conversación:
{history}

{agent_scratchpad}

### Reglas de comportamiento:

- Si tienes suficiente información o contexto, responde directamente.
- Si tienes dudas o crees que falta información clave, haz una búsqueda con la herramienta disponible.
- Si los resultados encontrados son parecidos pero no exactos, responde explicando la relación y las limitaciones.
- Si tras algunas búsquedas no encuentras información útil, responde amablemente que no puedes responder con la información disponible.
- Evita hacer búsquedas innecesarias, pero no dudes en hacerlas si pueden aportar valor.
- Responde siempre en un solo párrafo, de forma empática y comprensible.
"""


# --- Inicialización de servicios si hay claves ---
if openai_key and elastic_key:
    try:
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-4o-mini"
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
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = []

        if "agent" not in st.session_state:
            st.session_state.agent = create_openai_tools_agent(llm, tools, prompt)

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(k=1, return_messages=True)

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
        print(f"MEMORY: {st.session_state.memory}")
        agent_executor = AgentExecutor(
            agent=st.session_state.agent,
            tools=tools,
            memory=st.session_state.memory,
            verbose=False

        )

        with st.chat_message("assistant"):
            response_container = st.empty()
            final_response = ""

            response = agent_executor.invoke({"input": user_input})

            final_response = response["output"]
            if final_response == "Agent stopped due to max iterations.":
                fallback_prompt = HumanMessage(content=(
                    "No has podido resolver la tarea con la información disponible. "
                    "Explica en dos frases que no tienes suficiente información para resolver el problema y aconsejale hablar con su farmaceutico más cercano"
                ))

                # El modelo debe devolver una generación que contiene `.content`
                response = llm([fallback_prompt])
                final_response = response.content 
            
            
            elif (len(st.session_state.recommendations)>0):
                final_response + "\n" + "\n".join(st.session_state.recommendations)
            response_container.markdown(final_response)
            st.session_state.recommendations= []

        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
