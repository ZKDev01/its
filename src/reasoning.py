import re
import json
import math
import random
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from langchain_ollama import OllamaLLM

class ReasoningStrategy(Enum):
  "Estrategias de razonamiento disponibles"
  ANALYTICAL = "analytical"
  CREATIVE = "creative" 
  CRITICAL = "critical"
  EXPLORATORY = "exploratory"

@dataclass
class ThoughtNode:
  "Nodo que representa un fragmento de pensamiento en el árbol de razonamiento"
  topic:str
  thought: str
  reasoning_strategy: ReasoningStrategy
  depth: int
  parent: Optional['ThoughtNode'] = None
  children: List['ThoughtNode'] = None
  visits: int = 0
  value: float = 0.0
  confidence: float = 0.0
  
  def __post_init__(self):
    if self.children is None:
      self.children = []

  def is_fully_expanded(self, max_children:int) -> bool:
    "Verifica si el nodo está completamente expandido"
    return len(self.children) >= max_children
  
  def get_path_to_root(self) -> List['ThoughtNode']:
    "Obtiene el camino desde este nodo hasta la raíz"
    path = []
    current = self
    while current is not None:
      path.append(current)
      current = current.parent
    return list(reversed(path))
  
  def get_reasoning_chain(self) -> str:
    "Obtiene la cadena de razonamiento completa hasta este nodo"
    path = self.get_path_to_root()
    chain = []
    for i, node in enumerate(path):
      chain.append(f"Nivel {i}: [{node.reasoning_strategy.value}] {node.thought}")
    return "\n".join(chain)



strategy_prompt_for_analytical:str = """ 
Eres un experto analista que responde en español con rigor académico.
Contexto de razonamiento desarrollado previamente: {reasoning_context}

Aplicando un ENFOQUE ANALÍTICO riguroso, profundiza en este tema desarrollando:
- Descomposición sistemática del problema en componentes fundamentales
- Análisis detallado de relaciones causa-efecto y dependencias
- Identificación de patrones, tendencias y correlaciones significativas  
- Evaluación crítica de evidencia, datos y fuentes disponibles
- Estructuración lógica de argumentos y conclusiones

Desarrolla un análisis académico detallado, estructurado y fundamentado en español:
"""

strategy_prompt_for_creative:str = """ 
Eres un pensador creativo experto que responde en español con originalidad.
Contexto de razonamiento desarrollado previamente: {reasoning_context}

Aplicando un ENFOQUE CREATIVO innovador, explora este tema desarrollando:
- Pensamiento lateral con analogías, metáforas y conexiones inesperadas
- Perspectivas innovadoras que desafíen paradigmas convencionales
- Escenarios hipotéticos y modelos alternativos de comprensión
- Síntesis creativa entre disciplinas y dominios de conocimiento
- Soluciones originales y enfoques no tradicionales

Genera ideas creativas, originales y bien fundamentadas en español:
"""

strategy_prompt_for_critical:str = """ 
Eres un evaluador crítico experto que responde en español con objetividad.
Contexto de razonamiento desarrollado previamente: {reasoning_context}

Aplicando un ENFOQUE CRÍTICO riguroso, evalúa este tema desarrollando:
- Cuestionamiento sistemático de suposiciones y premisas fundamentales
- Identificación de sesgos, limitaciones y puntos ciegos potenciales
- Evaluación equilibrada de fortalezas, debilidades y riesgos
- Consideración de perspectivas alternativas y contraargumentos
- Análisis de implicaciones éticas, sociales y prácticas

Proporciona una evaluación crítica rigurosa, objetiva y bien argumentada en español:
"""

strategy_prompt_for_exploratory:str = """ 
Eres un investigador exploratorio experto que responde en español con curiosidad científica
Contexto de razonamiento desarrollado previamente: {reasoning_context}

Aplicando un ENFOQUE EXPLORATORIO sistemático, examina este tema desarrollando:
- Investigación multidireccional de aspectos inexplorados
- Formulación de preguntas abiertas y problemas de investigación
- Exploración de implicaciones a corto, medio y largo plazo
- Descubrimiento de nuevas dimensiones y conexiones interdisciplinarias
- Identificación de oportunidades de desarrollo futuro

Explora direcciones prometedoras de investigación con fundamento científico en español:
"""


GEMMA = "gemma3:1b"             # ollama gemma3:1b
DEEPSEEK = "deepseek-r1:1.5b"   # ollama deepseek-r1:1.5b

class LLMReasoning: 
  "Componente de razonamiento basado en MCTS para LLMs"
  
  def __init__(self, 
        main_model:str = GEMMA,
        critical_model:str = DEEPSEEK,
        temperature:float = 0.7,
        max_children:int = 3,
        exploration_weight:float = 1.41) -> None:
    """Inicializa el componente de razonamiento

    Args:
        main_model (str, optional): nombre del modelo de lenguaje local a utilizar. Defaults to GEMMA="gemma3:1b".
        critical_model (str, optional): nombre del modelo de lenguaje local a utilizar como crítico. Defaults to DEEPSEEK.
        temperature (float, optional): temperatura para la generación. Defaults to 0.7.
        max_children (int, optional): máximo número de hijos por nodo. Defaults to 3.
        exploration_weight (float, optional): peso de exploración para UCB. Defaults to 1.41.
    """
    self.main_llm = OllamaLLM(model=main_model, temperature=temperature)
    self.critical_llm = OllamaLLM(model=critical_model, temperature=0.9)
    self.max_children = max_children
    self.exploration_weight = exploration_weight
    self.reasoning_strategies = list(ReasoningStrategy)
    
    #TODO add retriever
  
  def expand_thought(self, parent_node:ThoughtNode, strategy:ReasoningStrategy) -> str:
    "Expande un pensamiento utilizando una estrategia específica"
    reasoning_context = parent_node.get_reasoning_chain()
    
    #TODO retriever.get_next_content
    
    strategy_prompts = {
      ReasoningStrategy.ANALYTICAL:  strategy_prompt_for_analytical,
      ReasoningStrategy.CREATIVE:    strategy_prompt_for_creative,
      ReasoningStrategy.CRITICAL:    strategy_prompt_for_critical,
      ReasoningStrategy.EXPLORATORY: strategy_prompt_for_exploratory
    }
    
    prompt = strategy_prompts[strategy].format(reasoning_context=reasoning_context)
    return self.main_llm.invoke(prompt)
  
  def evaluate_thought_quality(self, node:ThoughtNode) -> Dict[str,float]:
    "Evalúa la calidad de un pensamiento"
    prompt = f"""Eres un evaluador académico experto:
    Tema Original: {node.topic}
    
    Pensamiento a evaluar: {node.thought}
    
    Evalúa este pensamiento de manera rigurosa en las siguientes dimensiones usando las categorías: PÉSIMO, MALO, BUENO, EXCELENTE:
    
    1. RELEVANCIA: ¿Qué tan directamente relacionado está con el tema original? ¿Aporta valor específico?
    2. PROFUNDIDAD: ¿Qué tan profundo y detallado es el análisis presentado? ¿Demuestra comprensión compleja?
    3. ORIGINALIDAD: ¿Qué tan original e innovadora es la perspectiva ofrecida? 
    4. CLARIDAD: ¿Qué tan claro, comprensible y bien estructurado está el pensamiento?
    5. UTILIDAD: ¿Qué tan útil es este análisis para comprender mejor el tema original?
    
    INSTRUCCIONES DE FORMATO:
    {{
      "relevancia": pesimo/malo/bueno/excelente
      "profundidad": pesimo/malo/bueno/excelente
      "originalidad": pesimo/malo/bueno/excelente
      "claridad": pesimo/malo/bueno/excelente
      "utilidad": pesimo/malo/bueno/excelente
    }}
    
    Proporciona únicamente las evaluaciones categóricas en el formato exacto indicado y para cada evaluación solo se puede indicar una única categoría (pesimo/malo/bueno/excelente)
    """
    # generar respuesta por LLM crítico
    response = self.critical_llm.invoke(prompt).lower()
    
    # mapeo de conversión de categorías a valores numéricos
    category_mapping = {
      "pesimo": 0, 
      "malo": 1,
      "bueno": 2,
      "excelente": 3
    }
    
    # Buscar el primer bloque que parezca un JSON en la respuesta
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
      try:
        scores = json.loads(json_match.group(0).replace('\n', '').replace('\r', '').replace(' ', ''))
        for key in scores:
          scores[key] = category_mapping.get(str(scores[key]), 1.0)
      except Exception:
        scores = {}
    else:
      scores = {}
    
    metrics = ['relevancia', 'profundidad', 'originalidad', 'claridad', 'utilidad']
    # Asegurar que cada métrica esté presente en scores, si no, asignar 1.0 por defecto
    for metric in metrics:
      if metric not in scores:
        scores[metric] = 1.0    
    
    return scores
  
  def calculate_ucb(self, node:ThoughtNode, parent_visits:int) -> float:
    "Calcula el valor UCB para selección de nodos"
    if node.visits == 0: return float('inf')
    exploitation = node.value / node.visits
    exploration  = self.exploration_weight * math.sqrt((2 * math.log(parent_visits)) / node.visits)
    return exploitation + exploration
  
  def select_best_child(self, node:ThoughtNode) -> ThoughtNode:
    "Selecciona el mejor hijo usando UCB"
    if not node.children: return node 
    best_child = max(
      node.children, 
      key=lambda child: self.calculate_ucb(child,node.visits)
    )
    return best_child
  
  def select_node(self, root:ThoughtNode, max_depth:int) -> ThoughtNode:
    "Selecciona un nodo para expansión usando la política de selección"
    current = root 
    while ( current.is_fully_expanded(self.max_children) and 
            current.children and 
            current.depth < max_depth ):
      current = self.select_best_child(current)
    
    return current
  
  def expand_node(self, node:ThoughtNode, max_depth:int) -> Optional[ThoughtNode]:
    "Expande un nodo creando nuevos hijos"
    if node.depth >= max_depth or node.is_fully_expanded(self.max_children): return None
    
    # seleccionar estrategias de razonamiento
    available_strategies = [s for s in self.reasoning_strategies]
    strategy = random.choice(available_strategies)
    # Generar nuevo pensamiento
    new_thought = self.expand_thought(node, strategy)
    
    # Crear nuevo nodo hijo
    child_node = ThoughtNode(
      topic=node.topic,
      thought=new_thought,
      reasoning_strategy=strategy,
      depth=node.depth + 1,
      parent=node
    )
    
    node.children.append(child_node)
    return child_node
  
  def simulate(self, node:ThoughtNode) -> float:
    "Simula el valor de un nodo evaluando su calidad"
    scores = self.evaluate_thought_quality(node)
    total_sum = sum(scores.values())
    n_metrics = len(scores)
    avg_score = total_sum / n_metrics if n_metrics > 0 else 0.0
    final_score = avg_score / 3.0
    return max(min(final_score,1.0),0,0)
    
  def backpropagate(self, node:ThoughtNode, reward:float) -> None:
    "Propaga la recompensa hacia arriba en el árbol"
    current = node
    while current is not None:
      current.visits += 1
      current.value += reward
      current = current.parent
  
  def build_reasoning_tree(self, topic:str, max_depth:int=3, iterations:int=10, verbose:bool=False) -> ThoughtNode:
    """Construye un árbol de razonamiento usando MCTS

    Args:
        topic (str): tema sobre el cual razonar
        max_depth (int, optional): profundidad máxima del árbol. Defaults to 3.
        iterations (int, optional): número de iteraciones. Defaults to 10.

    Returns:
        ThoughtNode: nodo raíz del árbol de razonamiento construido
    """
    # generar pensamiento inicial
    #TODO optimizar el prompt de pensamiento inicial
    initial_thoughts = topic 
    if verbose: print(initial_thoughts)
    
    # crear nodo raíz
    root = ThoughtNode(
      topic=topic,
      thought=initial_thoughts[0],
      reasoning_strategy=ReasoningStrategy.EXPLORATORY,
      depth=0
    )
    
    for i in range(iterations):
      if verbose: print(f"Iteración {i+1}/{iterations}")
      
      # selección
      selected_node = self.select_node(root,max_depth)
      if verbose: print(f"Nodo seleccionado (depth={selected_node.depth})")
      
      # expansión
      expanded_node = self.expand_node(selected_node, max_depth)
      if expanded_node is None: expanded_node = selected_node
      if verbose: print(f"Nodo expandido: {expanded_node.reasoning_strategy.value}")
      
      # simulación
      reward = self.simulate(expanded_node)
      if verbose: print(f"Recompensa: {reward}")
      
      # backpropagation
      self.backpropagate(expanded_node, reward)
    
    return root 
  
  def _get_leaf_node(self, node:ThoughtNode) -> List[ThoughtNode]:
    if not node.children:
      return [node]
    leaves = []
    for child in node.children:
      leaves.extend(self._get_leaf_node(child))
    return leaves
  
  def get_best_reasoning_chains(self, root:ThoughtNode, n_chains:int=3) -> List[List[ThoughtNode]]:
    "Obtener las mejores cadenas de razonamiento del árbol"
    leaf_nodes = self._get_leaf_node(root)
    
    # ordenar por valor promedio 
    leaf_nodes.sort(key=lambda node:node.value/max(node.visits,1), reverse=True)
    
    # ordenar las mejores cadenas
    best_chains = []
    for leaf in leaf_nodes[:n_chains]:
      chain = leaf.get_path_to_root()
      best_chains.append(chain)
    
    return best_chains


def print_node(node:ThoughtNode, indent:int=0, max_depth:int=None):
  if max_depth is not None and node.depth > max_depth: return
  
  prefix = " " * indent
  avg_value = node.value / max(node.visits, 1)
  
  print(f"{prefix} |- Nivel {node.depth} [{node.reasoning_strategy.value}] "
        f"(número de visitas: {node.visits}, valor: {avg_value})")
  print(f"{prefix} {node.thought[:100]}{'...' if len(node.thought) > 100 else ''}")
  
  for child in node.children: print_node(child, indent + 1)

def print_reasoning_tree(root:ThoughtNode, max_depth:int=None):
  "Imprime el árbol de razonamiento de forma legible"
  print(f"===> Árbol de Razonamiento: {root.topic} <===")
  print_node(root)

def test() -> None:
  topic = "La importancia de las matrices en la inteligencia artificial"
  
  reasoning_component = LLMReasoning(
    main_model = "gemma3:1b",
    temperature = 0.7,
    max_children = 2,
    exploration_weight = 1.41
  )

  reasoning_tree = reasoning_component.build_reasoning_tree(
    topic=topic,
    max_depth=3,
    iterations=10,
    verbose=True
  )
  print_reasoning_tree(reasoning_tree)
  best_chains = reasoning_component.get_best_reasoning_chains(reasoning_tree, n_chains=3)

  print("\n=== Mejores cadenas de razonamiento ===")
  for i, chain in enumerate(best_chains, 1):
    print(f"\n --- Cadena {i} ---")
    for node in chain:
      avg_value = node.value / max(node.visits, 1)
      print(f"Nivel {node.depth} [{node.reasoning_strategy.value}] (valor: {avg_value})")
      print(f"Pensamiento: {node.thought}")


if __name__ == "__main__":
  test()