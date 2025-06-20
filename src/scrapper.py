import os
import json
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import (PlaywrightURLLoader, PyPDFLoader, WebBaseLoader, SeleniumURLLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
to answer the question. If you don't know the answer, just say that you don't know. 


Question: {question}
Context: {context}
Answer:
"""

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = None

model = OllamaLLM(model="llama3.2")

def load_dataset(config_path):

    def load_config():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Archivo de configuración no encontrado: {config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Error de formato en el archivo JSON: {config_path}")
            return {}
        
    documents = []
    config = load_config()

    # web_sources = None # Descomentar para ignorar las fuentes web
    web_sources = config.get("web_sources", []) #Descomentar para cargar las fuentes web

    #Variante para cargar todas las urls de una pasada
    def one_pass_load():
        if web_sources: 
            urls = [source["url"] for source in web_sources]
            try:
                online_loader = PlaywrightURLLoader( urls=urls, headless=True, continue_on_failure=True)
                web_docs = online_loader.load()
                documents.extend(web_docs)
                print(f"✅ Cargadas {len(web_sources)} fuentes web")
            except Exception as e:
                print(f"❌ Error cargando fuentes web: {str(e)}")

    #Variante para cargar las urls de una en una
    def one_by_one_load():
        if web_sources:  
            success_count = 0
            error_count = 0
            
            loader = PlaywrightURLLoader(None, headless=True, continue_on_failure=False)
            
            for source in web_sources:
                print(f"loading {success_count + error_count + 1}/{len(web_sources)}")
                url = source["url"]
                loader.urls = [url]

                try:
                    web_doc = loader.load()
                    documents.extend(web_doc)
                    success_count += 1
                except Exception as e:
                    print(f"❌ Error cargando {url}: {str(e)}")
                    error_count += 1
            
            print(f"\n✅ Cargadas con éxito: {success_count} fuentes web")
            print(f"❌ Errores: {error_count} fuentes web")
        
    one_pass_load()

    local_sources = config.get("local_sources", [])
    if local_sources:
        file_count = 0
        for source in local_sources:
            file_path = source.get("path", "")
            if not os.path.exists(file_path):
                print(f"⚠️ Archivo local no encontrado: {file_path}")
                continue

            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    local_docs = loader.load()
                else:
                    continue  # Saltar archivos no soportados

                documents.extend(local_docs)
                file_count += 1
            except Exception as e:
                print(f"Error cargando {file_path}: {e}")

        print(f"📂 Cargados {file_count} archivos locales")
    
    print(f"📚 Total documentos cargados: {len(documents)}")
    return documents


def split_documments(documents):
    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True
    )

    data = text_splitter.split_documents(documents)
    return data


def index_docs(documents):
    print(f"Indexando {len(documents)} documentos...")
    global vector_store
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Indexación completada")

def retrieve_docs(query):
    return vector_store.similarity_search(query)


def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question" : question, "context" : context})



raw_data = load_dataset("./config/sources.json")
chunked_data = split_documments(raw_data)
index_docs(chunked_data)

print("¡Preguntame algo!")
while (question := input("> ")) != "":
    retrieve_data  = retrieve_docs(question)

    context = "\n\n".join([doc.page_content[:500] for doc in retrieve_data])
    answer = answer_question(question, context)
    print(answer)