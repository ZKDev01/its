from typing import List,Dict 
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

GEMMA = "gemma3:1b"             # ollama gemma3:1b
DEEPSEEK = "deepseek-r1:1.5b"   # ollama deepseek-r1:1.5b

class Chatbot:
  def __init__(self,base_prompt:str, model_name:str=GEMMA,temperature:float=0.8):
    # optimizar base_prompt 
    self.base_prompt:str = base_prompt

    self.chat_prompt = ChatPromptTemplate.from_messages(
      [
        ('system', f"{self.base_prompt}"),
        MessagesPlaceholder(variable_name='memory'),
        ('human', '{input}')
      ]
    )
    
    # generate llm-chain
    self.llm:OllamaLLM = OllamaLLM(
      model=model_name,temperature=temperature
    )
    self.memory:List = []
    self.__chatbot = self.chat_prompt | self.llm
  
  def __call__(self, query:str): # *args, **kwds
    response = self.__chatbot.invoke( 
      { 
        'input': HumanMessage(query), 
        'memory': self.memory 
      } 
    )
    self.memory.append(HumanMessage(content=query))
    self.memory.append(AIMessage(content=response))
    
    return response




def main():
  queries = [
    "¿Qué es un número complejo?",
    "¿Qué tipo de operaciones tiene?",
    "¿Dime un proyecto que utilice esto?"
  ]
  prompt = """ 
  Eres un asistente de inteligencia artificial capaz de dar respuestas 
  razonando antes con la conversion tenida con el usuario
  Lenguaje de respuesta: Español
  """
  chatbot = Chatbot(prompt)
  r = []
  for query in queries:
    rt = chatbot(query)
    r.append(rt)
    print(rt)
    print("\n\n===============\n\n")  
  

if __name__ == "__main__":
  main()