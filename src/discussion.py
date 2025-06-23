import copy
import random
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from langchain_ollama import OllamaLLM
from langchain_core.messages import AIMessage 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

GEMMA = "gemma3:1b"             # ollama gemma3:1b
DEEPSEEK = "deepseek-r1:1.5b"   # ollama deepseek-r1:1.5b

class DiscussionRole(Enum):
  TUTOR = "tutor"
  OPPONENT = "opponent"

@dataclass
class Strategy:
  "Representa una estrategia de debate/discusión"
  credibility: float          # credibilidad (0.0 - 1.0)
  logical_reasoning: float    # razonamiento lógico (0.0 - 1.0)
  fitness: float = 0.0
  
  def __post_init__(self):
    # Normalizar valores entre 0 y 1
    self.credibility = max(0.0, min(1.0, self.credibility))
    self.logical_reasoning = max(0.0, min(1.0, self.logical_reasoning))

def categorize_credibility_value(_value:float) -> str:
  "Convierte un valor númerico de credibilidad (0.0-1.0) a categoría descriptiva"
  if _value < 0.3: return "baja"
  if _value < 0.7: return "estándar/normal"
  if _value < 1.0: return "alta"

def categorize_logical_reasoning_value(_value:float) -> str:
  "Convierte un valor númerico de razonamiento lógico (0.0-1.0) a categoría descriptiva"
  if _value < 0.6: return "baja"
  if _value < 1.0: return "alta"

def transform_strategy_to_categories(strategy:Strategy) -> Dict:
  """Transforma una instancia de Strategy a categorías descriptivas

  Args:
      strategy (Strategy): Instancia de Strategy

  Returns:
      Dict: Diccionario con las categorías de credibilidad y razonamiento lógico
  """
  return {
    "credibility": categorize_credibility_value(strategy.credibility),
    "logical_reasoning": categorize_logical_reasoning_value(strategy.logical_reasoning),
    "fitness": strategy.fitness
  }

class GeneticAlgorithm:
  "Implementa el algoritmo genético para evolucionar estrategias"
    
  def __init__(self, population_size:int=20, mutation_rate:float=0.1, alpha_balance:float = 0.3):      
    self.population_size:int = population_size if population_size > 1 else 10
    self.mutation_rate:float = mutation_rate if 0 <= mutation_rate and mutation_rate <= 1 else 0.1 
    self.alpha_balance:float = alpha_balance if 0 <= alpha_balance and alpha_balance <= 1 else 0.3
    
  def initialize_population(self) -> List[Strategy]:
    "Crea población inicial de estrategias"
    population = []
    for _ in range(self.population_size):
      strategy = Strategy(
        credibility=random.random(),
        logical_reasoning=random.random()
      )
      population.append(strategy)
    return population
    
  def evaluate_fitness(self, strategy:Strategy, debate_result:float) -> float:
    "Evalúa la aptitud de una estrategia basada en resultados del debate"
    # Combina balance de elementos con resultado del debate
    balance_score = 1.0 - abs(0.5 - strategy.credibility/2) - abs(0.5 - strategy.logical_reasoning/2) 
    return (balance_score*self.alpha_balance) + (debate_result*(1-self.alpha_balance))
    
  def tournament_selection(self, population:List[Strategy], tournament_size:int=3) -> Strategy:
    "Selección por torneo"
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda s: s.fitness)
    
  def crossover(self, parent1: Strategy, parent2: Strategy) -> Tuple[Strategy, Strategy]:
    "Cruce uniforme entre dos estrategias"
    child1 = Strategy(
      credibility = parent1.credibility if random.random() < 0.5 else parent2.credibility,
      logical_reasoning = parent1.logical_reasoning if random.random() < 0.5 else parent2.logical_reasoning,
    )
    child2 = Strategy(
      credibility = parent2.credibility if random.random() < 0.5 else parent1.credibility,
      logical_reasoning = parent2.logical_reasoning if random.random() < 0.5 else parent1.logical_reasoning,
    )
    return child1, child2
    
  def mutate(self, strategy:Strategy) -> Strategy:
    "Aplica mutación a una estrategia"
    mutated = copy.deepcopy(strategy)
    if random.random() < self.mutation_rate: mutated.credibility += random.gauss(0, 0.1)
    if random.random() < self.mutation_rate: mutated.logical_reasoning += random.gauss(0, 0.1)
    
    # Normalizar después de mutación
    mutated.credibility = max(0.0, min(1.0, mutated.credibility))
    mutated.logical_reasoning = max(0.0, min(1.0, mutated.logical_reasoning))
    
    return mutated
    
  def evolve(self, population:List[Strategy]) -> List[Strategy]:
    "Evoluciona una generación de la población"
    # Ordenar por fitness
    population.sort(key=lambda s: s.fitness, reverse=True)
    
    # Mantener élite (20% mejores)
    elite_size = self.population_size // 5
    new_population = population[:elite_size]
    
    # Generar resto de población
    while len(new_population) < self.population_size:
      parent1 = self.tournament_selection(population)
      parent2 = self.tournament_selection(population)
            
      child1, child2 = self.crossover(parent1, parent2)
      child1 = self.mutate(child1)
      child2 = self.mutate(child2)
            
      new_population.extend([child1, child2])
        
    return new_population[:self.population_size]

@dataclass
class DiscussionMove:
  "Representa un movimiento en el debate"
  role: DiscussionRole
  content: str
  strategy: Strategy
  score: float = 0.0

@dataclass
class DiscussionState:
  "Estado actual del debate"
  topic: str
  moves: List[DiscussionMove]
  current_turn: DiscussionRole
  iteration: int

class MinimaxSearch:
  "Implementa búsqueda adversarial minimax para planificación de movimientos"

  def __init__(self, max_depth:int=3):
    self.max_depth:int = max_depth
    
  def evaluate_move(self, move: DiscussionMove, state: DiscussionState) -> float:
    "Evalúa la calidad de un movimiento"
    strategy = move.strategy
    # Puntuación base según rol
    base_score = 0.5
    # Bonificación por balance de estrategia
    balance_bonus = 1.0 - abs(strategy.credibility - strategy.logical_reasoning)
    
    # Penalización por repetición de patrones
    repetition_penalty = 0.0
    recent_moves = state.moves[-3:] if len(state.moves) >= 3 else state.moves
    if recent_moves:
      avg_credibility = sum(m.strategy.credibility for m in recent_moves) / len(recent_moves)
      if abs(strategy.credibility - avg_credibility) < 0.1: repetition_penalty += 0.2
    
    return base_score + (balance_bonus * 0.3) - repetition_penalty
    
  def generate_possible_moves(self, state: DiscussionState, role: DiscussionRole, strategies: List[Strategy]) -> List[DiscussionMove]:
    "Genera posibles movimientos para un rol dado"
    moves = []
    
    # Seleccionar mejores estrategias
    top_strategies = sorted(strategies, key=lambda s: s.fitness, reverse=True)[:5]
    
    for strategy in top_strategies:
      # trasformar en categorías descriptivas
      tmp_category = transform_strategy_to_categories(strategy)
      credibility = tmp_category["credibility"]
      logical_reasoning = tmp_category["logical_reasoning"]
      
      # Contenido básico basado en rol y estrategia
      if role == DiscussionRole.TUTOR:
        content = f"Explicación con credibilidad={credibility}, razonamiento lógico={logical_reasoning}"
      else:
        content = f"Contraargumento con credibilidad={logical_reasoning}, razonamiento lógico={logical_reasoning}"
      
      move = DiscussionMove(role=role, content=content, strategy=strategy)
      move.score = self.evaluate_move(move, state)
      moves.append(move)
        
    return moves
  
  def minimax(self, 
        state:DiscussionState, 
        depth:int, 
        maximizing_player:bool, 
        tutor_strategies:List[Strategy], 
        opponent_strategies:List[Strategy] ) -> Tuple[float,Optional[DiscussionMove]]:
    """Algoritmo minimax para seleccionar mejor movimiento

    Args:
        state (DiscussionState): _description_
        depth (int): _description_
        maximizing_player (bool): _description_
        tutor_strategies (List[Strategy]): _description_
        opponent_strategies (List[Strategy]): _description_

    Returns:
        Tuple[float,Optional[DiscussionMove]]: _description_
    """
        
    if depth == 0 or len(state.moves) >= 20:  # Condición de parada
      return sum(move.score for move in state.moves if move.role == DiscussionRole.TUTOR) - sum(move.score for move in state.moves if move.role == DiscussionRole.OPPONENT), None
    
    current_role = DiscussionRole.TUTOR if maximizing_player else DiscussionRole.OPPONENT
    strategies = tutor_strategies if maximizing_player else opponent_strategies
    
    possible_moves = self.generate_possible_moves(state, current_role, strategies)
    
    if maximizing_player:
      max_eval = float('-inf')
      best_move = None
      
      for move in possible_moves:
        new_state = copy.deepcopy(state)
        new_state.moves.append(move)
        new_state.current_turn = DiscussionRole.OPPONENT
        new_state.iteration += 1
        
        eval_score, _ = self.minimax(new_state, depth - 1, False, tutor_strategies, opponent_strategies)
        
        if eval_score > max_eval:
          max_eval = eval_score
          best_move = move
      
      return max_eval, best_move
    else:
      min_eval = float('inf')
      best_move = None
      
      for move in possible_moves:
        new_state = copy.deepcopy(state)
        new_state.moves.append(move)
        new_state.current_turn = DiscussionRole.TUTOR
        new_state.iteration += 1
        
        eval_score, _ = self.minimax(new_state, depth - 1, True, tutor_strategies, opponent_strategies)
        
        if eval_score < min_eval:
          min_eval = eval_score
          best_move = move
            
      return min_eval, best_move

class Discussion:
  "Clase principal que integra LLMs, GA y búsqueda adversarial"
  
  def __init__(self, 
        tutor_model:str = GEMMA, 
        oponente_model:str = DEEPSEEK, 
        t_tutor_model:float = 0.7, 
        t_opponent_model:float = 0.7 ) -> None:
    # Inicializar LLMs
    self.tutor_llm = OllamaLLM(model=tutor_model, temperature=t_tutor_model)
    self.opponent_llm = OllamaLLM(model=oponente_model, temperature=t_opponent_model)
    
    # Inicializar componentes
    self.genetic_algorithm = GeneticAlgorithm()
    self.minimax_search = MinimaxSearch()
    
    # Poblaciones de estrategias
    self.tutor_strategies = self.genetic_algorithm.initialize_population()
    self.opponent_strategies = self.genetic_algorithm.initialize_population()
    
    # Plantillas de prompts
    self.tutor_prompt = ChatPromptTemplate.from_messages([
      ('system', """Eres un tutor experto que debe explicar conceptos de manera didáctica.
      Tu estrategia actual enfatiza:
      - Credibilidad: {credibility} 
      - Razonamiento Lógico: {logical_reasoning} 
      
      Adapta tu respuesta según estas categorías. Responde en español."""),
      MessagesPlaceholder(variable_name='memory'),
      ('human', 'Tema: {topic}\nÚltimo argumento del oponente: {opponent_arg}\nPuedes utilizar el siguiente conocimiento del tema: {context}\nExplica y defiende tu posición:')
    ])
    
    self.opponent_prompt = ChatPromptTemplate.from_messages([
      ('system', """Eres un oponente crítico que busca contraargumentos y debilidades.
      Tu estrategia actual enfatiza:
      - Credibilidad: {credibility}   
      - Razonamiento Lógico: {logical_reasoning} 
      
      Adapta tu respuesta según estas categorías. Responde en español."""),
      MessagesPlaceholder(variable_name='memory'),
      ('human', 'Tema: {topic}\nArgumento del tutor: {tutor_arg}\nPuedes utilizar el siguiente conocimiento del tema: {context}\nPresenta contraargumentos:')
    ])
    
    self.memory:List = []
    
  def generate_llm_response(self, role: DiscussionRole, topic: str, opponent_arg: str, strategy: Strategy) -> str:
    "Genera respuesta usando LLM con estrategia específica"
    
    # trasformar en categorías descriptivas
    tmp_category = transform_strategy_to_categories(strategy)
    credibility = tmp_category["credibility"]
    logical_reasoning = tmp_category["logical_reasoning"]
    
    # TODO search context (Agent Retriever)
    # adicionar contexto según número de iteraciones => k_iterations = k_different_context / k_different_reasoning_paths(_subset)
    context = ''
    
    if role == DiscussionRole.TUTOR:
      chain = self.tutor_prompt | self.tutor_llm
      response = chain.invoke({
        'topic': topic,
        'opponent_arg': opponent_arg or "Ninguno (primera iteración)",
        'credibility': credibility,
        'logical_reasoning': logical_reasoning,
        'memory': self.memory,
        'context': context
      })
    else:
      chain = self.opponent_prompt | self.opponent_llm
      response = chain.invoke({
        'topic': topic,
        'tutor_arg': opponent_arg or "Ninguno",
        'credibility': credibility,
        'logical_reasoning': logical_reasoning,
        'memory': self.memory,
        'context': context
      })
    
    self.memory.append(AIMessage(response))
    
    return response
    
  def run_discuss(self, 
        topic:str, 
        k_iterations:int,
        verbose:bool = True ) -> Dict:
    """Ejecuta la discusión con K iteraciones
  
    Args:
        topic (str): _description_
        k_iterations (int): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        Dict: _description_
    """
    # Inicializar estado del debate
    state = DiscussionState(
      topic=topic,
      moves=[],
      current_turn=DiscussionRole.TUTOR,
      iteration=0
    )
    
    self.memory = []
    
    if verbose: print(f"Iniciando discusión: {topic}")
    
    results = []
    for iteration in range(k_iterations):
      if verbose: print(f"\n===========> Iteración {iteration + 1}/{k_iterations}")
      
      # TURNO DEL TUTOR
      if state.current_turn == DiscussionRole.TUTOR or iteration == 0:
        # Usar minimax para seleccionar mejor estrategia
        _, best_move = self.minimax_search.minimax(
          state, depth=2, maximizing_player=True,
          tutor_strategies=self.tutor_strategies,
          opponent_strategies=self.opponent_strategies
        )
        
        if best_move:
          strategy = best_move.strategy
        else:
          # Fallback: usar mejor estrategia actual
          strategy = max(self.tutor_strategies, key=lambda s: s.fitness)
        
        # Obtener último argumento del oponente
        last_opponent_arg = ""
        for move in reversed(state.moves):
          if move.role == DiscussionRole.OPPONENT:
            last_opponent_arg = move.content
            break
        
        # Generar respuesta del tutor
        tutor_response = self.generate_llm_response(
          DiscussionRole.TUTOR, topic, last_opponent_arg, strategy
        )

        tutor_move = DiscussionMove(
          role=DiscussionRole.TUTOR,
          content=tutor_response,
          strategy=strategy
        )
        tutor_move.score = self.minimax_search.evaluate_move(tutor_move, state)
        
        state.moves.append(tutor_move)
        
        # trasformar en categorías descriptivas
        tmp_category = transform_strategy_to_categories(strategy)
        credibility = tmp_category["credibility"]
        logical_reasoning = tmp_category["logical_reasoning"]

        if verbose: print(f"{DiscussionRole.TUTOR} (Strategy: C={credibility}, L={logical_reasoning}):\n{tutor_response}")
        
        results.append({
          'iteration': iteration + 1,
          'role': DiscussionRole.TUTOR,
          'strategy': strategy,
          'content': tutor_response,
          'score': tutor_move.score
        })
        
      # TURNO DEL OPONENTE
      if len(state.moves) > 0 and state.moves[-1].role == DiscussionRole.TUTOR:
        # Usar minimax para seleccionar mejor estrategia del oponente
        _, best_move = self.minimax_search.minimax(
          state, depth=2, maximizing_player=False,
          tutor_strategies=self.tutor_strategies,
          opponent_strategies=self.opponent_strategies
        )
        
        if best_move:
          strategy = best_move.strategy
        else:
          # Fallback: usar mejor estrategia actual
          strategy = max(self.opponent_strategies, key=lambda s: s.fitness)
        
        # Generar respuesta del oponente
        opponent_response = self.generate_llm_response(
          DiscussionRole.OPPONENT, topic, state.moves[-1].content, strategy
        )
        
        opponent_move = DiscussionMove(
          role=DiscussionRole.OPPONENT,
          content=opponent_response,
          strategy=strategy
        )
        opponent_move.score = self.minimax_search.evaluate_move(opponent_move, state)
        
        state.moves.append(opponent_move)
        
        # trasformar en categorías descriptivas
        tmp_category = transform_strategy_to_categories(strategy)
        credibility = tmp_category["credibility"]
        logical_reasoning = tmp_category["logical_reasoning"]

        if verbose: print(f"{DiscussionRole.OPPONENT} (Strategy: C={credibility}, L={logical_reasoning}):\n{opponent_response}")
        
        results.append({
          'iteration': iteration + 1,
          'role': DiscussionRole.OPPONENT,
          'strategy': strategy,
          'content': opponent_response,
          'score': opponent_move.score
        })
            
      # Evolucionar estrategias en cada iteración 
      self._evolve_strategies(state)
      if verbose: print(f"Estrategias evolucionadas (Iteración {iteration + 1})")
      
      state.iteration = iteration + 1
      if verbose: print("\n" + "-"*60)
    
    # Calcular resultados finales
    results = self._calculate_results(state, results)
    
    if verbose:
      print("\nResultados finales")
      print(f"Score {DiscussionRole.TUTOR}: {results['tutor_score']}")
      print(f"Score {DiscussionRole.OPPONENT}: {results['opponent_score']}")
    return results
    
  def _evolve_strategies(self, state: DiscussionState):
    """Evoluciona las estrategias basándose en el rendimiento del debate"""
    
    # Calcular fitness para estrategias del tutor
    for strategy in self.tutor_strategies:
      tutor_moves = [m for m in state.moves if m.role == DiscussionRole.TUTOR and m.strategy == strategy]
      if tutor_moves:
        avg_score = sum(m.score for m in tutor_moves) / len(tutor_moves)
        strategy.fitness = self.genetic_algorithm.evaluate_fitness(strategy, avg_score)
        
    # Calcular fitness para estrategias del oponente
    for strategy in self.opponent_strategies:
      opponent_moves = [m for m in state.moves if m.role == DiscussionRole.OPPONENT and m.strategy == strategy]
      if opponent_moves:
        avg_score = sum(m.score for m in opponent_moves) / len(opponent_moves)
        strategy.fitness = self.genetic_algorithm.evaluate_fitness(strategy, avg_score)
        
    # Evolucionar poblaciones
    self.tutor_strategies = self.genetic_algorithm.evolve(self.tutor_strategies)
    self.opponent_strategies = self.genetic_algorithm.evolve(self.opponent_strategies)
    
  def _calculate_results(self, state: DiscussionState, history: List[Dict]) -> Dict:
    "Calcula los resultados finales del debate"
    
    tutor_score = sum(m.score for m in state.moves if m.role == DiscussionRole.TUTOR)
    opponent_score = sum(m.score for m in state.moves if m.role == DiscussionRole.OPPONENT)
    
    return {
      'topic': state.topic,
      'total_iterations': state.iteration,
      'tutor_score': tutor_score,
      'opponent_score': opponent_score,
      'discussion_history': history,
      'final_strategies': {
        'tutor_best': max(self.tutor_strategies, key=lambda s: s.fitness),
        'opponent_best': max(self.opponent_strategies, key=lambda s: s.fitness)
      }
    }
  
  def generate_final_response(self, discussion_results:Dict) -> str:
    """Genera una respuesta final consolidada del topic después del debate

    Args:
        discussion_results (Dict): resultados de la discusión/debate obtenidos de `run_discuss()`

    Returns:
        str: respuesta final consolidada que integra todo el conocimiento de la discusión
    """
    topic = discussion_results["topic"]
    history = discussion_results["discussion_history"]
    best_tutor_strategy:Strategy = discussion_results["final_strategies"]['tutor_best']
    best_opponent_strategy:Strategy = discussion_results["final_strategies"]['opponent_best']
    
    # Extraer argumentos clave del tutor y contraargumentos del oponente
    tutor_args = [entry['content'] for entry in history if entry['role'] == DiscussionRole.TUTOR]
    opponent_args = [entry['content'] for entry in history if entry['role'] == DiscussionRole.OPPONENT]
    
    #!DELETE HERE
    print("\n\n\n============")
    print(tutor_args)
    print("============\n\n\n")
    print(opponent_args)
    print("============\n\n\n")
    
    # Crear resumen de argumentos principales
    tutor_summary = "\n".join([f"• {arg[:200]}..." if len(arg) > 200 else f"• {arg}" for arg in tutor_args])
    opponent_summary = "\n".join([f"• {arg[:200]}..." if len(arg) > 200 else f"• {arg}" for arg in opponent_args])
        
    # Crear estrategia híbrida óptima combinando las mejores características
    hybrid_strategy = Strategy(
      credibility=(best_tutor_strategy.credibility + best_opponent_strategy.credibility)/2,
      logical_reasoning=(best_tutor_strategy.logical_reasoning + best_opponent_strategy.logical_reasoning)/2
    )
    
    # Crear prompt para respuesta final
    final_prompt = ChatPromptTemplate.from_messages(
      [
        ('system', """Eres un experto síntesis que debe generar una respuesta final completa y equilibrada sobre un tema.
            
        Has observado un debate completo entre un tutor y un oponente. Tu tarea es crear una respuesta definitiva que:
        1. Integre los mejores argumentos de ambas partes
        2. Aborde las objeciones y contraargumentos presentados
        3. Proporcione una explicación comprehensiva y balanceada
        4. Mantenga un enfoque pedagógico y accesible
            
        Tu estrategia de respuesta debe enfatizar:
        - Credibilidad: {credibility} 
        - Razonamiento Lógico: {logical_reasoning}
        
        Genera una respuesta en español que sea definitiva, educativa y que resuelva el topic original."""),
        ('human', """TEMA ORIGINAL: {topic}

        ARGUMENTOS DEL TUTOR:
        {tutor_arguments}

        CONTRAARGUMENTOS/OBJECIONES DEL OPONENTE:
        {opponent_arguments}

        Genera una respuesta final que sintetice todo este conocimiento en una explicación comprehensiva del tema original.""")
      ]
    )
    
    # Usar el modelo específico para síntesis 
    synthesis_llm = self.tutor_llm
    
    # Generar respuesta final
    chain = final_prompt | synthesis_llm
    
    # trasformar los valores de la estrategia híbrida en categorías descriptivas 
    tmp_category = transform_strategy_to_categories(hybrid_strategy)
    credibility = tmp_category["credibility"]
    logical_reasoning = tmp_category["logical_reasoning"]
    
    final_response = chain.invoke({
      'topic': topic,
      'tutor_arguments': tutor_summary,
      'opponent_arguments': opponent_summary,
      'credibility': hybrid_strategy.credibility,
      'logical_reasoning': hybrid_strategy.logical_reasoning
    })
    
    return final_response





def test():
  discussion = Discussion()

  topic = "¿Qué son las matrices?"
  k_iterations = 2

  results = discussion.run_discuss(topic, k_iterations, verbose=False)
  final_response = discussion.generate_final_response(results)

  print("\n\nFinal RESPONSE")
  print(final_response)

if __name__ == "__main__":
  test()