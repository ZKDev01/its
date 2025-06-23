from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


embeddings = OllamaEmbeddings(
  model = "mxbai-embed-large"
)

