""" Creacion de vectorstore y graphDB a partir de únicamente documentos
1. Procesar dataset 
1.1. Obtener por cada archivo markdown los chunks 

# Vectorstore
1.1.1. Obtener de cada ckunks tres componentes <C,S,QA>
      - C: chunk original
      - S: resumen de lo que dice el chunk
      - QA: preguntas y respuesta utilizando el mismo chunk

# Graph DB
1.1.2. Extraer de cada chunk las entidades y relaciones
    - Entidades: primero se analizan las entidades
    - Relaciones: luego se pregunta al LLM cuales entidades estan relacionadas y como

1.2. Salvar cada resultado en DB diferentes
"""
from typing import List
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.graph_retriever import GraphDB
from src.information_extraction import extract_triples
from src.scrapper import split_documments, load_dataset


def create_domain():
  # 1. Cargar los documentos
  documents = load_dataset()

  # 1.1. Procesar los documentos a markdown
  chunks:List[Document] = split_documments(documents)

  embeddings = OllamaEmbeddings(
    model = "mxbai-embed-large"
  )
  vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embeddings,
    persist_directory='./chroma'
  )
  uuids = [str(uuid4()) for _ in range(len(chunks))]
  vector_store.add_documents(
    documents=chunks, 
    ids=uuids
  )
  
  for chunk in chunks:
    triples = extract_triples(chunk.page_content)
    for e in triples: print(e)



if __name__ == "__main__":
  create_domain()