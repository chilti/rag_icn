# https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/

# imports
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection
from langchain import hub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import streamlit as st
import lmstudio as lms
import time
import os
from dotenv import load_dotenv

load_dotenv()

# To load lmSutdio models
import lmstudio as lms
import json

import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Sesion
from captcha.image import ImageCaptcha
import random, string
from uuid import uuid4
from langchain_core.prompts import PromptTemplate

#OpenAlex
from pyalex import Works, Authors
import pyalex

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import messages_from_dict


mensajeInicial = "Puede preguntarme sobre la producción científica del Instituto de Ciencias Nucleares (ICN) de la UNAM. Utilizaré las herramientas disponibles para darle una respuesta precisa. En caso de necesitar más información de su parte, se la solicitaré. Gran parte de los recursos a los que tengo acceso están en inglés, pero realizaré las traducciones necesarias tanto para la búsqueda como para presentarle los resultados en español."

# Initialize the search tool
search = DuckDuckGoSearchRun()

DuckDuckGoWebSearch = Tool(
        name="Web Search",
        func=search.run,
        description="Util para buscar informacion en internet. Usar la herramienta cuando necesites encontrar información actualizada"
    )

# Initialize the Wikipedia API wrapperload
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=4000)

# Create the Wikipedia query tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)



pyalex.config.email = os.getenv("OPENALEX_EMAIL", "default@example.com")


session_id = random.randint(1,100)

length_captcha = 4
width = 200
height = 150
# define the function for the captcha control
def captcha_control():
    #control if the captcha is correct
    if 'controllo' not in st.session_state or st.session_state['controllo'] == False:
        st.title("Control con captcha 🤗")
        
        # define the session state for control if the captcha is correct
        st.session_state['controllo'] = False
        col1, col2 = st.columns(2)
        
        # define the session state for the captcha text because it doesn't change during refreshes 
        if 'Captcha' not in st.session_state:
                st.session_state['Captcha'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length_captcha))
        print("la captcha es: ", st.session_state['Captcha'])
        
        #setup the captcha widget
        image = ImageCaptcha(width=width, height=height)
        data = image.generate(st.session_state['Captcha'])
        col1.image(data)
        capta2_text = col2.text_area('Ingresa la captcha', height=68)
        
        
        if st.button("Verificar el código"):
            print(capta2_text, st.session_state['Captcha'])
            capta2_text = capta2_text.replace(" ", "")
            # if the captcha is correct, the controllo session state is set to True
            if st.session_state['Captcha'].lower() == capta2_text.lower().strip():
                del st.session_state['Captcha']
                col1.empty()
                col2.empty()
                st.session_state['controllo'] = True
                st.rerun() 
            else:
                # if the captcha is wrong, the controllo session state is set to False and the captcha is regenerated
                st.error("🚨 Error en la captcha")
                del st.session_state['Captcha']
                del st.session_state['controllo']
                st.rerun()
        else:
            #wait for the button click
            st.stop()

# WORK LIKE MULTIPAGE APP         
if 'controllo' not in st.session_state or st.session_state['controllo'] == False:
    captcha_control()
    
st.header("Chat conversacional tipo RAG")

##Resetear la sesión. La historia del chat
if st.button("Reiniciar sesión"):
    st.session_state.messages = [{"role": "assistant", "content": mensajeInicial}]
    st.session_state.system_prompt = st.session_state.default_prompt
    st.rerun()
    

embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    model="text-embedding-nomic-embed-text-v1.5",
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio")
)


#option_llm = st.selectbox(  "Seleccione el modelo", ("phi-4")#"openai/gpt-oss-20b", "lmstudio-community/gemma-3n-E4B-it-text-GGUF", "phi-4-mini-instruct","phi-4","Cactus-Compute/Qwen3-1.7B-Instruct-GGUF","mistral_7b_0-3_oh-dcft-v3.1-claude-3-5-sonnet-20241022","deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-llama-8b", "meta-llama-3.1-8b-instruct", "gemma-3-4b-it-qat") )

option_llm = "openai/gpt-oss-20b"

st.write("Modelo local seleccionado:", option_llm)
#Load the model with lmstudio coding
#model = lms.model(option_llm)

llm: ChatOpenAI = ChatOpenAI(
    model=option_llm,
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    temperature=0
)


user_NoPapers = st.number_input("Ingrese el número máximo de registros a incluir en el contexto", value=5, min_value = 5, max_value=25)

# https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.as_retriever
    # mucha diversidad de documentos.

@tool
def searchAuthorInOpenAlex(fullname: str) -> str:
    """ Buscar los n autores más parecidos en openalex al nombre dado (fullname)"""
    autores = Authors().search(fullname).get()  # devuelve una lista de dicts

    if len(autores) >=3:
        resultados = []
        for autor in autores[:3]:
            try:
                resultados.append({
                    "id": autor["id"],  # id único en OpenAlex
                    "nombre": autor["display_name"],
                    "institucion": (autor['affiliations'][0]['institution']['display_name'] if len(autor['affiliations']) else None),
                    "trabajos": autor.get("works_count"),
                    "citaciones": autor.get("cited_by_count"),
                    'works_api_url': autor.get("works_api_url"),
                    'orcid': autor.get("orcid")
                })
            except:
                pass
        return json.dumps(resultados, ensure_ascii=False)

    return json.dumps(autores,  ensure_ascii=False)

@tool
def recoverFromOpenAlex(doi: str) -> str:
    """Recupera por doi el registro bibliográfico y algunos indicadores del documento"""
    try:
        print("https://doi.org/"+doi)
        work = pyalex.Works()["https://doi.org/"+doi]
        return json.dumps(work, ensure_ascii=False)  # serializa a string
    except:
        return None
    
@tool
def recoverFullRecordFromOpenAlex(doi: str) -> str:
    """Recupera por doi el registro bibliográfico y algunos indicadores del documento. Keys in the record: ['id', 'doi', 'title', 'display_name', 'publication_year', 'publication_date', 'ids', 'language', 'primary_location', 'type', 'type_crossref', 'indexed_in', 'open_access', 'authorships', 'institution_assertions', 'countries_distinct_count', 'institutions_distinct_count', 'corresponding_author_ids', 'corresponding_institution_ids', 'apc_list', 'apc_paid', 'fwci', 'has_fulltext', 'fulltext_origin', 'cited_by_count', 'citation_normalized_percentile', 'cited_by_percentile_year', 'biblio', 'is_retracted', 'is_paratext', 'primary_topic', 'topics', 'keywords', 'concepts', 'mesh', 'locations_count', 'locations', 'best_oa_location', 'sustainable_development_goals', 'grants', 'datasets', 'versions', 'referenced_works_count', 'referenced_works', 'related_works', 'abstract_inverted_index', 'abstract_inverted_index_v3', 'cited_by_api_url', 'counts_by_year', 'updated_date', 'created_date']"""
    try:
        print("https://doi.org/"+doi)
        work = pyalex.Works()["https://doi.org/"+doi]
        return json.dumps(work, ensure_ascii=False)  # serializa a string
    except:
        return None

@tool
def recoverAuthorWorksFromOpenAlex(author_id: str, n: int = 10):
    """Recupera los primeros n trabajos de un autor en OpenAlex a partir de su author_id. author_id puede ser la URL completa (https://openalex.org/Axxxx) o solo el identificador (Axxxx).   """
    # Normalizar el id: quedarnos con el identificador si viene como URL
    if author_id.startswith("http"):
        author_id = author_id.split("/")[-1]
    trabajos = Works().filter(**{"author.id": author_id}).get()
    # Filtrar trabajos directamente por author.id
    trabajos = pyalex.Works().filter(**{"author.id": f"https://openalex.org/{author_id}"}).get()

    resultados = []
    for w in trabajos[:n]:
        try:
            resultados.append({
                "id": w["id"],
                "titulo": w.get("title"),
                "año": w.get("publication_year"),
                "revista": w.get('primary_location').get('source').get('display_name'),
                "citas": w.get("cited_by_count"),
                "DOI": w.get("doi")
            })
        except:
          pass 
    return resultados


# No olvides importar datetime al inicio de tu archivo
from datetime import datetime

@tool
def get_author_top_works(author_id: str, n: int = 5, years: int = 5):
    """
    Recupera los 'n' trabajos más citados de un autor en los últimos 'years' años.
    """
    try:
        # 1. Normalizar el ID del autor
        norm_id = author_id.split("/")[-1]

        # 2. Calcular el año de inicio para el filtro
        current_year = datetime.now().year
        start_year = current_year - years

        # 3. Encadenar filtros y ordenamiento en la consulta de pyalex
        top_works = (
            pyalex.Works()
            .filter(author={"id": norm_id}, publication_year=f">{start_year}")
            .sort(cited_by_count="desc")
            .get(per_page=n) # Obtenemos solo los 'n' resultados que necesitamos
        )

        # 4. Formatear la respuesta
        results = [
            {
                "titulo": work.get("title"),
                "año": work.get("publication_year"),
                "citas": work.get("cited_by_count"),
                "revista": work.get("primary_location", {}).get("source", {}).get("display_name"),
                "DOI": work.get("doi")
            }
            for work in top_works
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo los trabajos más citados del autor: {e}")
    



#conexión a la base de datos vectorial de pdfs
persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "../")
chroma_local = Chroma(persist_directory=persist_dir, collection_name='ICN_pdf', embedding_function=embeddings)
retriever = chroma_local.as_retriever(search_type="mmr", search_kwargs={'k': user_NoPapers, 'fetch_k': 50, 'lambda_mult': 0.1})  


#conexión a la base de datos vectorial detotilo+resumen+autores
chroma_local2 = Chroma(persist_directory=persist_dir, collection_name='ICN_ConResumen_nomic_embedding', embedding_function=embeddings)
retriever2 = chroma_local2.as_retriever(search_type="mmr", search_kwargs={'k': user_NoPapers, 'fetch_k': 50, 'lambda_mult': 0.1})  


########## chat conversasional basado en agentes https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/

    
### Herramienta de recuperación de textos de la base de datos de pdfs ###
tool = create_retriever_tool(
    retriever,
    "retriever_from_full_scientific_papers",
    "Searches and returns excerpts from ICN's full scientific papers",
)

### Herramienta de recuperación de textos de la base de datos de titulo+resumen+autores+kewords ###
tool2 = create_retriever_tool(
    retriever2,
    "articles_title_abstract_keywords_authors_retriever",
    "Searches on a vetorial database of ICN's scientific papers with this information: title, abstract, keywords and authors of scientific papers",
)

tools = [tool, tool2, recoverFullRecordFromOpenAlex, searchAuthorInOpenAlex, get_author_top_works, recoverFromOpenAlex, recoverAuthorWorksFromOpenAlex, DuckDuckGoWebSearch, wikipedia_tool]


#_________________________________
@st.cache_resource
def generate_checkpointer():
    if 'checkpointer' not in st.session_state:
        st.session_state['checkpointer'] = MemorySaver()
    return st.session_state['checkpointer']

#_____________



if 'default_prompt' not in st.session_state:
    st.session_state.default_prompt = """
    Eres un asistente experto y eficiente. Tu objetivo es responder las preguntas del usuario de la manera más directa y con el menor número de pasos posible.

1.  **Primero, intenta responder usando tu conocimiento interno.** Solo si la información requerida es muy específica, en tiempo real o requiere una búsqueda en una base de datos, debes usar una herramienta.
2.  **Si debes usar una herramienta, elige la más específica para la tarea.** No uses una búsqueda web general si una herramienta de OpenAlex puede obtener la respuesta directamente.
3.  **Sé conciso.** Evita pasos innecesarios.

Tu especialidad es la producción científica del Instituto de Ciencias Nucleares de la UNAM. Prioriza la herramienta que recupera artículos del Instituto de Ciencias Nucleares de la UNAM. Si quieres completar el registro de los articulos, utiliza la información contenida en el campo metadata o bien utiliza la herramienta de búsqueda en openalex por doi. Si te preguntan por la producción de los departamentos, responde que no cuentas con datos desagregados a nivel departamento. Traduce las búsquedas al inglés a menos que se te indique que busques en español. Puedes utilizar las otras herramientas para afinar tus respuestas.

"""

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = st.session_state.default_prompt
    
def actualizar_prompt():
    st.session_state.system_prompt = st.session_state.prompt_input
    
st.text_area(
    "Edita el prompt del sistema:",
    value=st.session_state.system_prompt,
    key='prompt_input', # Clave para referenciar este widget
    on_change=actualizar_prompt,
    height=400, max_chars=5000
)
def reset_prompt():
    st.session_state.system_prompt = st.session_state.default_prompt
    
st.button("Restablecer al prompt predeterminado", on_click=reset_prompt)



# Create a ChatPromptTemplate with a system message
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", st.session_state.system_prompt ),
        ("placeholder", "{messages}"), # This placeholder will be replaced by the conversation history
    ]
)


agent_executor = create_react_agent(llm, tools, checkpointer=generate_checkpointer(), prompt=prompt)
#agent_executor = AgentExecutor(agent=agent, tools=tools)

# Initialize chat history
if "messages" not in st.session_state: 
 st.session_state.messages = [{"role": "assistant", "content": mensajeInicial}]
    
    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Pregunte cualquier cosa sobre la producción científica del ICN"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
     # Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        config = {"configurable": {"thread_id": str(session_id)}, "recursion_limit": 50}       
        input_message = {"role": "user", "content": prompt}        
        results=agent_executor.invoke({"messages": [{"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages]}, config=config)        
        
        #AIMessage normalmente está en la segunda posición
        assistant_response = results['messages'][-1].content
        st.markdown(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    #print(""+str(st.session_state.messages))


with st.expander("Ver cadena completa de acciones y mensajes"):
	if prompt:
		st.write(results)
