import re 
import sys 
import json
from io import StringIO
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_ollama import (
  OllamaLLM,
  ChatOllama,
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
from src.ga_ils_scheduling_systems import run_scheduling, Solution

GEMMA = "gemma3:1b"             # ollama gemma3:1b
DEEPSEEK = "deepseek-r1:1.5b"   # ollama deepseek-r1:1.5b

DEFAULT_SYSTEM_PROMPT = """ 
Eres un asistente especializado capaz de responder preguntas sencillas
"""





class TaskType(Enum):
  TASK_PLANNER = "planificador de tareas"
  TEXT_GENERATION = "generación mejorada de argumentos"
  REASONING = "razonamiento"
  QA_GENERATION = "generación de preguntas y respuestas"

CLASSIFICATION_PROMPT = """ 
Analiza la siguiente descripción de tarea y determina cuál de estos tipos de tarea es el más adecuado:

TIPOS DE TAREA DISPONIBLES:
1. TASK_PLANNER (planificador de tareas) - Para crear planes, rutas de aprendizaje, organizar actividades, etc.
2. TEXT_GENERATION (generación mejorada de argumentos) - Para crear contenido textual, escribir argumentos, explicaciones, etc.
3. REASONING (razonamiento) - Para analizar, justificar, deducir, resolver problemas complejos, etc.
4. QA_GENERATION (generación de preguntas y respuestas) - Para generar preguntas, respuestas, exámenes, evaluaciones, etc.

DESCRIPCIÓN DE LA TAREA:
"{description}"

INSTRUCCIONES:
- Responde ÚNICAMENTE con el nombre del tipo de tarea (por ejemplo: TASK_PLANNER)
- No incluyas explicaciones adicionales
- Si hay ambigüedad, elige el tipo más específico y relevante
- Considera las palabras clave y el contexto de la descripción

RESPUESTA:
"""

def get_task_type_by_description(description:str) -> TaskType:
  "Clasifica el tipo de task según los tipos de task disponibles"
  llm = Ollama(model=GEMMA, temperature=0.3)
  response = llm(CLASSIFICATION_PROMPT.format(description=description)).strip().upper()
  task_type_mapping: Dict[str, TaskType] = {
    "TASK_PLANNER": TaskType.TASK_PLANNER,
    "TEXT_GENERATION": TaskType.TEXT_GENERATION,
    "REASONING": TaskType.REASONING,
    "QA_GENERATION": TaskType.QA_GENERATION
  }
  
  if response in task_type_mapping:
    return task_type_mapping[response]
  
  return _classify_by_keywords(description)

def _classify_by_keywords(description: str, llm_response: str = "") -> TaskType:
  desc_lower = description.lower()
  response_lower = llm_response.lower()
  keywords: Dict[TaskType, List[str]] = {
    TaskType.TASK_PLANNER: [
      "plan", "planificar", "ruta de aprendizaje", "agenda", "organizar", "secuencia", "estrategia"
    ],
    TaskType.TEXT_GENERATION: [
      "escribir", "generar texto", "argumento", "redactar", "crear contenido", "explicar", "describir"
    ],
    TaskType.REASONING: [
      "razonar", "deducir", "analizar", "justificar", "demostrar", "resolver", "pensar", "argumentar"
    ],
    TaskType.QA_GENERATION: [
      "pregunta", "respuesta", "evaluar", "generar preguntas", "cuestionario", "test", "examen"
    ]
  }
  for task_type, words in keywords.items():
    for word in response_lower.split():
      if word in words:
        return task_type
  task_scores: Dict[TaskType, int] = {task_type: 0 for task_type in TaskType}
  for task_type, words in keywords.items():
    for word in words:
      if word in desc_lower:
        task_scores[task_type] += 1
  if max(task_scores.values()) > 0:
    return max(task_scores, key=task_scores.get)
  return TaskType.TEXT_GENERATION



class TaskStatus(Enum):
  PENDING = "pendiente"
  COMPLETED = "completado"
  FAILED = "fallido"

@dataclass
class Task:
  task_id:str 
  task_type:TaskType
  description:str
  result:Optional[Any] = None
  assigned_agent:Optional[str] = None
  status:TaskStatus = TaskStatus.PENDING
  error_message:Optional[str] = None

  def __init__(self, task_id:str, task_type:TaskType, description:str, result:Optional[Any]=None, assigned_agent:Optional[str]=None, status:TaskStatus=TaskStatus.PENDING, error_message:Optional[str]=None):
    self.task_id = task_id
    self.task_type = task_type
    self.description = description
    self.result = result
    self.assigned_agent = assigned_agent
    self.status = status
    self.error_message = error_message

  def assign_to_agent(self, agent_name:str):
    self.assigned_agent = agent_name

  def mark_as_completed(self, result:Any):
    self.status = TaskStatus.COMPLETED
    self.result = result

  def mark_as_failed(self, error_message:str):
    self.status = TaskStatus.FAILED
    self.error_message = error_message





class LLM(ABC):
  def __init__(self, model:str, api_key:str="", temperature:float=0.7):
    self.model = model
    self.api_key = api_key 
    self.temperature = temperature
  
  @abstractmethod
  def __call__(self, query:str) -> str:
    pass 

class Ollama(LLM):
  def __init__(self, model:str, api_key:str = "", temperature:float = 0.7, system_prompt:str = ""):
    super().__init__(model, api_key, temperature)
    
    try:
      self.llm = OllamaLLM(self.model, self.temperature)
    except Exception:
      self.llm = OllamaLLM(model=GEMMA, temperature=0.7)
    
    self.system_prompt:str = system_prompt if len(system_prompt) > 0 else DEFAULT_SYSTEM_PROMPT
    self.memory:List = []
    
    self.chat_prompt = ChatPromptTemplate.from_messages(
      [
        ('system', f'{self.system_prompt}'),
        MessagesPlaceholder(variable_name='memory'),
        ('human', '{query}')
      ]
    )
    self.chain = self.chat_prompt | self.llm  
  
  def __call__(self, query:str) -> str:
    response = self.chain.invoke(
      {
        "query": HumanMessage(query),
        "memory": self.memory
      }
    )
    self.memory.append(HumanMessage(content=query))
    self.memory.append(AIMessage(content=response))
    
    return response

class BaseResponse:
  """
  Clase base para manejar respuestas del sistema. 
  Permite almacenar información de éxito/error y datos adicionales
  """
  def __init__(self, success:bool, agent:str, result:Any=None, message:str="", **kwargs:Dict[str,Any]) -> None:
    """Inicializa la respuesta base

    Args:
      success (bool): indica si la operación fue exitosa.
      agent (str): nombre del agente que ejecutó la operación.
      result (Any): resultado de la operación (puede ser cualquier tipo)
      message (str): descripción de la operación realizada. Defaults to "".
    """
    self.success = success
    self.agent   = agent 
    self.result  = result
    self.message = message 
    self.extra_data = kwargs

  def get_info(self) -> Dict[str,Any]:
    """Obtiene la información de la respuesta

    Returns:
      Dict[str,Any]: Diccionario con información básica de la respuesta
    """
    info = {
      "success": self.success,
      "agent": self.agent,
      "result": self.result,
      "message": self.message
    }
    info.update(self.extra_data)
    return info
  
  def get_data(self, key:str, default:Any=None) -> Any:
    """Obtiene un dato específico de los datos extra.

    Args:
      key (str): clave del dato a obtener
      default (Any): valor por defecto si la clave no existe. Defaults to None.

    Returns:
      Any: valor del dato solicitado o el valor por defecto
    """
    return self.extra_data.get(key, default)
  
  def __str__(self) -> str:
    "Representación en string de la información"
    status = "SUCCESS" if self.success else "ERROR"
    return f"[{status}] {self.agent}: {self.message}"
  
  def __repr__(self) -> str:
    "Representación detallada de la respuesta"
    return f"BaseResponse(success={self.success}, agent='{self.agent}', message='{self.message}')"

class BaseAgent(ABC):
  """
  Clase base abstracta para todos los agentes del sistema.
  Define la interfaz común que deben implementar todos los agentes.
  """
  
  def __init__(self, name:str, description:str, capabilities:List[TaskType]) -> None:
    self.name:str = name 
    self.description:str = description
    self.capabilities:List[TaskType] = capabilities
  
  @abstractmethod
  def process(self, queries:List[str], results:List[Dict] = [], **kwargs:Dict[str,Any]) -> BaseResponse:
    "Método abstracto que debe implementar cada agente específico"
    pass 
  
  def can_handle_task(self, task_type:TaskType) -> bool:
    "Verifica si el agente puede manejar un tipo específico de tarea"
    return task_type in self.capabilities
  
  def get_info(self) -> Dict[str,str]:
    "Retorna información básica del agente"
    return {
      "name": self.name, 
      "description": self.description
    }



class TaskPlannerAgent(BaseAgent):
    def __init__(self, llm: LLM):
        super().__init__(
            name="TaskPlannerAgent",
            description="Genera planes, rutas de aprendizaje y organiza actividades.",
            capabilities=[TaskType.TASK_PLANNER]
        )
        self.llm = llm

    def process(self, queries: List[str], results: List[Dict] = [], **kwargs) -> BaseResponse:
        if not queries:
            return BaseResponse(False, self.name, None, "No se proporcionó consulta.")
        try:
            solution: Solution = run_scheduling()
            plan_text = f"Solución óptima de asignación de tareas:\n{solution}"
            # Devuelve también el objeto Solution en el campo extra_data
            return BaseResponse(
                True, 
                self.name, 
                plan_text, 
                "Plan generado exitosamente por el algoritmo de scheduling.",
                solution=solution
            )
        except Exception as e:
            return BaseResponse(False, self.name, None, f"Error en el algoritmo de scheduling: {str(e)}")

class TextGenAgent(BaseAgent):
  def __init__(self, llm: LLM, documents: List[str] = []):
    super().__init__(
      name="TextGenAgent",
      description="Genera texto y argumentos mejorados.",
      capabilities=[TaskType.TEXT_GENERATION]
    )
    self.llm = llm
    self.documents = documents

  def process(self, queries: List[str], results: List[Dict] = [], **kwargs) -> BaseResponse:
    "Procesa una consulta basándose en los documentos disponibles"
    if not queries:
      return BaseResponse(
        success=False,
        agent=self.name,
        result=None,
        message="No se encuentran 'queries' en los parámetros"
      )
    
    answers = []
    for query in queries:
      answer = self._process_query(query=query)
      answers.append(answer)
    
    return BaseResponse(
      success=True, 
      agent=self.name,
      result=answers,
      message="Procesamiento de consultas completado exitosamente"
    )
  
  def _process_query(self, query: str) -> str:
    "Procesa una consulta individual usando los documentos disponibles"
    enhanced_query = f""" 
    {query}
    ---
    Documentos disponibles:
    {self.documents}
    """
    response = self.llm(query=enhanced_query)
    return response

class ReasoningAgent(BaseAgent):
    def __init__(self, llm: LLM):
        super().__init__(
            name="ReasoningAgent",
            description="Realiza razonamientos, deducciones y análisis complejos.",
            capabilities=[TaskType.REASONING]
        )
        self.llm = llm

    def process(self, queries: List[str], results: List[Dict] = [], **kwargs) -> BaseResponse:
        if not queries:
            return BaseResponse(False, self.name, None, "No se proporcionó consulta.")
        reasoning_prompt = f"Resuelve y justifica el siguiente problema complejo:\n{queries[0]}"
        reasoning = self.llm(reasoning_prompt)
        return BaseResponse(True, self.name, reasoning, "Razonamiento realizado exitosamente.")

class QAGenerationAgent(BaseAgent):
    def __init__(self, llm: LLM):
        super().__init__(
            name="QAGenerationAgent",
            description="Genera preguntas y respuestas, exámenes y evaluaciones.",
            capabilities=[TaskType.QA_GENERATION]
        )
        self.llm = llm

    def process(self, queries: List[str], results: List[Dict] = [], **kwargs) -> BaseResponse:
        if not queries:
            return BaseResponse(False, self.name, None, "No se proporcionó consulta.")
        qa_prompt = f"Genera preguntas y respuestas relevantes sobre:\n{queries[0]}"
        qa = self.llm(qa_prompt)
        return BaseResponse(True, self.name, qa, "Preguntas y respuestas generadas exitosamente.")
        
prompt_AgentEngine = """
Tienes que elegir el agente más apropiado para responder la siguiente consulta.
Clasificada como: {task_type}

Agentes disponibles:
{agents_info}

Consulta del usuario: "{query}"

Responde ÚNICAMENTE con el nombre exacto del agente más apropiado (sin explicaciones adicionales).
"""

class TaskManager:
  def __init__(self, llm: LLM):
    self.task_queue: List[Task] = []
    self.agents: Dict[str, BaseAgent] = {}
    self.completed_tasks: List[Task] = []
    self.failed_tasks: List[Task] = []
    self.task_counter: int = 0
    self.llm: LLM = llm  # LLM para selección de agentes
  
  def add_agent(self, agent: BaseAgent) -> BaseResponse:
    "Registra un nuevo agente en el sistema"
    try:
      self.agents[agent.name] = agent
      return BaseResponse(
        success=True,
        agent=agent.name,
        result=None,
        message="Agente registrado exitosamente"
      )
    except Exception as e:
      return BaseResponse(
        success=False,
        agent=agent.name,
        result=None,
        message=f"Error al registrar agente: {str(e)}"
      )
  
  def add_task(self, task:Task) -> None:
    "Añade una nueva tarea en el sistema"
    self.task_queue.append(task)
  
  def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
    "Devuelve la información de todos los agentes disponibles"
    return {name: agent.get_info() for name, agent in self.agents.items()}
  
  def select_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
    "Selecciona el agente más apropiado para una tarea específica"
    if not self.agents:
      return None
    
    # Filtrar agentes que pueden manejar el tipo de tarea
    capable_agents = [
      agent for agent in self.agents.values() 
      if agent.can_handle_task(task.task_type)
    ]
    
    if not capable_agents:
      return None
    
    if len(capable_agents) == 1:
      return capable_agents[0]
    
    # Si hay múltiples agentes capaces, usar LLM para seleccionar el mejor
    agents_info = "\n".join([
      f"- {agent.name}: {agent.description} (Capacidades: {[cap.value for cap in agent.capabilities]})"
      for agent in capable_agents
    ])
    
    selection_prompt = prompt_AgentEngine.format(
      agents_info=agents_info,
      query=task.description,
      task_type=task.type.value
    ) 
    
    try:
      response = self.llm(selection_prompt).strip()
      
      # Buscar el agente por nombre en la respuesta
      for agent in capable_agents:
        if agent.name in response:
          return agent
      
      # Si no encuentra coincidencia exacta, devolver el primer agente capaz
      return capable_agents[0]
      
    except Exception:
      # En caso de error, devolver el primer agente capaz
      return capable_agents[0]
  
  def assign_task_to_agent(self, task: Task, agent: BaseAgent) -> bool:
    "Asigna una tarea específica a un agente"
    try:
      task.assign_to_agent(agent.name)
      return True
    except Exception:
      return False
  
  def execute_task(self, task: Task, results:List[Dict]) -> BaseResponse:
    "Ejecuta una tarea usando el agente asignado"
    if not task.assigned_agent:
      return BaseResponse(
        success=False,
        agent="TaskManager",
        result=None,
        message="Tarea no tiene agente asignado"
      )
    
    agent:BaseAgent = self.agents.get(task.assigned_agent)
    if not agent:
      return BaseResponse(
        success=False,
        agent="TaskManager",
        result=None,
        message=f"Agente {task.assigned_agent} no encontrado"
      )
    
    try:
      # Ejecutar la tarea con el agente
      response = agent.process(
        queries=[task.description],
        results=results
      )
      
      if response.success:
        task.mark_as_completed(response.result)
        self.completed_tasks.append(task)
      else:
        task.mark_as_failed(response.message)
        self.failed_tasks.append(task)
      
      return response
      
    except Exception as e:
      task.mark_as_failed(str(e))
      self.failed_tasks.append(task)
      return BaseResponse(
        success=False,
        agent=task.assigned_agent,
        result=None,
        message=f"Error al ejecutar tarea: {str(e)}"
      )
  
  def get_pending_tasks(self) -> List[Task]:
    "Obtiene todas las tareas pendientes"
    return [task for task in self.task_queue if task.status == TaskStatus.PENDING]
  
  def get_task_status(self) -> Dict[str, int]:
    "Obtiene un resumen del estado de las tareas"
    all_tasks = self.task_queue + self.completed_tasks + self.failed_tasks
    return {
      "pending": len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
      "in_progress": len([t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]),
      "completed": len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]),
      "failed": len([t for t in all_tasks if t.status == TaskStatus.FAILED])
    }





CLASSIFICATION_PROMPT = """ 
Analiza la siguiente descripción de tarea y determina cuál de estos tipos de tarea es el más adecuado:

TIPOS DE TAREA DISPONIBLES:
1. TASK_PLANNER (planificador de tareas) - Para crear planes, rutas de aprendizaje, organizar actividades, etc.
2. TEXT_GENERATION (generación mejorada de argumentos) - Para crear contenido textual, escribir argumentos, explicaciones, etc.
3. REASONING (razonamiento) - Para analizar, justificar, deducir, resolver problemas complejos, etc.
4. QA_GENERATION (generación de preguntas y respuestas) - Para generar preguntas, respuestas, exámenes, evaluaciones, etc.

DESCRIPCIÓN DE LA TAREA:
"{description}"

INSTRUCCIONES:
- Responde ÚNICAMENTE con el nombre del tipo de tarea (por ejemplo: TASK_PLANNER)
- No incluyas explicaciones adicionales
- Si hay ambigüedad, elige el tipo más específico y relevante
- Considera las palabras clave y el contexto de la descripción

RESPUESTA:
"""

def get_task_type_by_description(description:str) -> TaskType:
  "Clasifica el tipo de task según los tipos de task disponibles"
  llm = Ollama(model=GEMMA, temperature=0.3)
  response = llm(CLASSIFICATION_PROMPT.format(description=description)).strip().upper()
  task_type_mapping: Dict[str, TaskType] = {
    "TASK_PLANNER": TaskType.TASK_PLANNER,
    "TEXT_GENERATION": TaskType.TEXT_GENERATION,
    "REASONING": TaskType.REASONING,
    "QA_GENERATION": TaskType.QA_GENERATION
  }
  
  if response in task_type_mapping:
    return task_type_mapping[response]
  
  return _classify_by_keywords(description)

def _classify_by_keywords(description: str, llm_response: str = "") -> TaskType:
  desc_lower = description.lower()
  response_lower = llm_response.lower()
  keywords: Dict[TaskType, List[str]] = {
    TaskType.TASK_PLANNER: [
      "plan", "planificar", "ruta de aprendizaje", "agenda", "organizar", "secuencia", "estrategia"
    ],
    TaskType.TEXT_GENERATION: [
      "escribir", "generar texto", "argumento", "redactar", "crear contenido", "explicar", "describir"
    ],
    TaskType.REASONING: [
      "razonar", "deducir", "analizar", "justificar", "demostrar", "resolver", "pensar", "argumentar"
    ],
    TaskType.QA_GENERATION: [
      "pregunta", "respuesta", "evaluar", "generar preguntas", "cuestionario", "test", "examen"
    ]
  }
  for task_type, words in keywords.items():
    for word in response_lower.split():
      if word in words:
        return task_type
  task_scores: Dict[TaskType, int] = {task_type: 0 for task_type in TaskType}
  for task_type, words in keywords.items():
    for word in words:
      if word in desc_lower:
        task_scores[task_type] += 1
  if max(task_scores.values()) > 0:
    return max(task_scores, key=task_scores.get)
  return TaskType.TEXT_GENERATION



class AgentEngine:
  """
  Interfaz principal donde TaskManager y los agentes se comunican 
  para resolver problemas complejos
  """
  
  def __init__(self, llm: LLM):
    self.task_manager = TaskManager(llm)
    self.llm: LLM = llm
    self.execution_history: List[Dict[str, Any]] = []
  
  def register_agent(self, agent: BaseAgent) -> BaseResponse:
    """Registra un nuevo agente en el sistema"""
    return self.task_manager.add_agent(agent)
      
  def run_cycle(self, complex_problem: str) -> BaseResponse:
    """
    Ejecuta un ciclo completo del sistema para resolver un problema complejo:
    1. Clasifica el tipo de tarea usando classify_by_keywords.
    2. Selecciona el agente adecuado según el tipo de tarea.
    3. Ejecuta el agente y devuelve la respuesta.
    """
    try:
      task_type = get_task_type_by_description(complex_problem)
      task = Task(
        task_id="main",
        task_type=task_type,
        description=complex_problem
      )
      self.task_manager.add_task(task)

      selected_agent = self.task_manager.select_agent_for_task(task)
      if not selected_agent:
        task.mark_as_failed("No se encontró agente apropiado")
        return BaseResponse(
          success=False,
          agent="AgentEngine",
          result=None,
          message="No se encontró agente apropiado"
        )

      self.task_manager.assign_task_to_agent(task, selected_agent)

      execution_result = self.task_manager.execute_task(task, results=[])
      # Si el agente devuelve una Solution, también la guardamos en el historial
      self.execution_history.append({
        "problem": complex_problem,
        "task_type": task_type.value,
        "agent": selected_agent.name,
        "result": execution_result.result,
        "success": execution_result.success,
        "message": execution_result.message,
        "solution": execution_result.get_data("solution", None)
      })

      return execution_result

    except Exception as e:
      return BaseResponse(
        success=False,
        agent="AgentEngine",
        result=None,
        message=f"Error durante la ejecución del ciclo: {str(e)}"
      )
  
  def get_execution_history(self) -> List[Dict[str, Any]]:
    "Obtiene el historial de ejecuciones"
    return self.execution_history
  
  def get_system_status(self) -> Dict[str, Any]:
    "Obtiene el estado general del sistema"
    return {
      "registered_agents": len(self.task_manager.agents),
      "available_agents": list(self.task_manager.agents.keys()),
      "task_status": self.task_manager.get_task_status(),
      "execution_history_count": len(self.execution_history)
    }