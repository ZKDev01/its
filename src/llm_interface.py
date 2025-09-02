from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_ollama import (
  OllamaLLM,
  OllamaEmbeddings
)
from langchain_core.messages import (
  BaseMessage,
  AIMessage,
  HumanMessage  
)
from langchain_core.prompts import (
  ChatPromptTemplate,
  MessagesPlaceholder  
)
from traitlets import default

# CONSTANTES
## NOMBRE DE MODELOS
GEMMA = "gemma3:1b"             # ollama gemma3:1b
DEEPSEEK = "deepseek-r1:1.5b"   # ollama deepseek-r1:1.5b
## DEFAULT PROMPTS
DEFAULT_SYSTEM_PROMPT = """ 
Eres un asistente especializado capaz de responder preguntas sencillas
"""



class LLM(ABC):
  def __init__(self, 
      model:Optional[str]="", 
      api_key:Optional[str]="", 
      base_url:Optional[str]="", 
      temperature:Optional[float]=0.7
    ) -> None:
    """_summary_

    Args:
      model (str): _description_
      api_key (Optional[str], optional): _description_. Defaults to "".
      base_url (Optional[str], optional): _description_. Defaults to "".
      temperature (Optional[float], optional): _description_. Defaults to 0.7.
    """
    self.model = model 
    self.api_key = api_key 
    self.base_url = base_url
    self.temperature = temperature
  
  @abstractmethod
  def __call__(self, query:str, **kwds) -> str:
    pass 

class Ollama(LLM):
  ""
  
  def __init__(self, 
      model:Optional[str]="", 
      api_key:Optional[str]="", 
      base_url:Optional[str]="", 
      temperature:Optional[float]=0.7,
      system_prompt:Optional[str]=""
    ) -> None:
    super().__init__(model, api_key, base_url, temperature)
    self.system_prompt:str = system_prompt
    
    try:
      self.llm = OllamaLLM(self.model, self.temperature)
    except Exception:
      self.llm = OllamaLLM(model=GEMMA, temperature=0.7)
    
    self.system_prompt:str = system_prompt if len(system_prompt) > 0 else DEFAULT_SYSTEM_PROMPT 
    self.memory:List[BaseMessage] = []
    
    self.chat_prompt = ChatPromptTemplate.from_messages(
      [
        ('system', f'{self.system_prompt}'),
        MessagesPlaceholder(variable_name='memory'),
        ('human', '{query}')
      ]
    )
    self.chain = self.chat_prompt | self.llm
  
  
  def __call__(self, query, **kwds) -> str:
    """_summary_

    Args:
      query (_type_): _description_

    Returns:
      str: _description_
    """
    if not kwds.get("in_memory", None):
      response = self.llm.invoke(query)
      return response
    
    response = self.chain.invoke(
      {
        "query": HumanMessage(query),
        "memory": self.memory
      }
    )
    self.memory.append( HumanMessage(content=query) )
    self.memory.append( HumanMessage(content=query) )
    
    return response

class OnlineLLM(LLM):
  ""
  
  pass 




