import json 
import random 
from typing import List, Tuple, Set, Optional, Union
from dataclasses import dataclass, asdict

@dataclass
class Task:
  "Representación de una tarea con sus propiedades"
  id: int
  description: str
  duration: int         # d(k): duración en turnos
  priority: float       # p(k): prioridad
  load: float           # load(k): carga/intensidad
  flexibility: float    # flex(k): flexibilidad (0 es inflexible)
  valid_turns: Set[int] # w(k): turnos permitidos
  valid_days: Set[int]  # vd(k): días permitidos
  ideal_turns: Set[Tuple[int,int]] = None # turno ideal

@dataclass 
class Day:
  "Representación de un día con sus propiedades"
  id: int               
  energy: float         # energy(n): capacidad energética
  busy_slots: Set[int]  # busy-slots(n): turnos ocupados por eventos fijos

@dataclass
class Assignment:
  "Representación de una asignación de tarea a día y turno"
  task_id: int 
  day: int 
  start_turn: int 
  
  def __repr__(self):
    return f"(id={self.task_id}, day={self.day}, start_turn={self.start_turn})"
  def __str__(self):
    return self.__repr__()

class Solution:
  "Representa una solución completa del problema"
  def __init__(self, assignments: List[Assignment]):
    self.assignments = assignments
    self.assignment_map = {a.task_id: a for a in assignments}
    self.fitness = -float('inf')
    self.penalty = float('inf')
    self.is_penalty = True
  
  def get_assignment(self, task_id: int) -> Assignment:
    "Obtiene la asignación de una tarea específica"
    return self.assignment_map[task_id]
  
  def copy(self) -> 'Solution':
    "Crea una copia de la solución"
    new_assignments = [Assignment(a.task_id, a.day, a.start_turn) for a in self.assignments]
    return Solution(new_assignments)

  def __repr__(self):
    output = "\n".join([str(a) for a in self.assignments]) 
    return output
  def __str__(self):
    return self.__repr__()



@dataclass
class ParamsConfig:
  # parámetros de la función objetivo
  gamma1:float        # peso para adaptabilidad
  gamma2:float        # peso para robusteza
  delta:float         # peso para margin 
  lambda_param:float  # peso para flex-dev
  penalty_threshold:float # U - umbral de penalización
  # pesos para restricciones blandas
  beta_window:Union[int,float]      
  beta_valid_days:Union[int,float]   

class TaskSchedulerProblem:
  "Define el problema de asignación de tareas"
  def __init__(self, tasks: List[Task], days: List[Day], turns_per_day: int, params:ParamsConfig):
    self.tasks = tasks
    self.days = days
    self.turns_per_day = turns_per_day
    self.task_map = {t.id: t for t in tasks}
    self.day_map = {d.id: d for d in days}
    
    # Parámetros de la función objetivo  
    self.gamma1 = params.gamma1
    self.gamma2 = params.gamma2
    self.delta = params.delta
    self.lambda_param = params.lambda_param
    self.penalty_threshold = params.penalty_threshold
    # Pesos para restricciones blandas
    self.beta_window = params.beta_window
    self.beta_valid_days = params.beta_valid_days

  def is_feasible(self, solution: Solution) -> bool:
    "Verifica si una solución cumple las restricciones duras"
    occupied_slots = {}  # (day, turn) -> task_id
    
    for assignment in solution.assignments:
      task = self.task_map[assignment.task_id]
      day_obj = self.day_map[assignment.day]
      
      # Verificar duración de la tarea
      end_turn = assignment.start_turn + task.duration - 1
      if end_turn >= self.turns_per_day:
        return False
      
      # Verificar solapamiento y eventos fijos
      for turn in range(assignment.start_turn, end_turn + 1):
        slot = (assignment.day, turn)
        
        # Verificar eventos fijos
        if turn in day_obj.busy_slots:
          return False
        
        # Verificar solapamiento
        if slot in occupied_slots:
          return False
        
        occupied_slots[slot] = assignment.task_id
    
    return True

  def calculate_penalty(self, solution: Solution) -> float:
    "Calcula la penalización por restricciones blandas"
    penalty = 0.0
    
    for assignment in solution.assignments:
      task = self.task_map[assignment.task_id]
      
      # Penalización por ventana temporal
      if assignment.start_turn not in task.valid_turns:
        penalty += self.beta_window
      
      # Penalización por días válidos
      if assignment.day not in task.valid_days:
        penalty += self.beta_valid_days
    
    return penalty

  def calculate_adaptability(self, solution: Solution) -> float:
    "Calcula el componente de adaptabilidad E(x)"
    daily_loads = {}
    
    # Calcular carga por día
    for assignment in solution.assignments:
      task = self.task_map[assignment.task_id]
      if assignment.day not in daily_loads:
        daily_loads[assignment.day] = 0.0
      daily_loads[assignment.day] += task.load
    
    # Calcular desviación cuadrática
    adaptability = 0.0
    for day in self.days:
      actual_load = daily_loads.get(day.id, 0.0)
      if day.energy > 0:
        deviation = (actual_load - day.energy) / day.energy
        adaptability += deviation ** 2
      
    return -adaptability  # Negativo porque queremos minimizar

  def calculate_margin(self, solution: Solution) -> float:
    "Calcula el margen mínimo entre tareas"
    daily_tasks = {}
    
    # Agrupar tareas por día
    for assignment in solution.assignments:
      if assignment.day not in daily_tasks:
        daily_tasks[assignment.day] = []
      daily_tasks[assignment.day].append(assignment)
    
    min_margin = float('inf')
    
    for _, assignments in daily_tasks.items():
      if not assignments:
        continue
      # Ordenar por turno de inicio
      assignments.sort(key=lambda a: a.start_turn)
      # Calcular gaps
      gaps = []
      # Gap antes de la primera tarea
      if assignments:
        gaps.append(assignments[0].start_turn)
      
      # Gaps entre tareas consecutivas
      for i in range(len(assignments) - 1):
        curr_task = self.task_map[assignments[i].task_id]
        curr_end = assignments[i].start_turn + curr_task.duration
        next_start = assignments[i + 1].start_turn
        gap = next_start - curr_end
        gaps.append(gap)
      
      # Gap después de la última tarea
      if assignments:
        last_task = self.task_map[assignments[-1].task_id]
        last_end = assignments[-1].start_turn + last_task.duration
        gaps.append(self.turns_per_day - last_end)
      
      # Encontrar el gap mínimo del día
      if gaps:
        day_min_gap = min(gaps)
        min_margin = min(min_margin, day_min_gap)
    
    return min_margin if min_margin != float('inf') else 0.0

  def calculate_flex_deviation(self, solution: Solution) -> float:
    "Calcula la desviación de flexibilidad"
    flex_dev = 0.0
    
    for assignment in solution.assignments:
      task = self.task_map[assignment.task_id]
      if task.flexibility > 0 and task.ideal_turns is not None:
        deviations = []
        for ideal_day, ideal_turn in task.ideal_turns:
          x = self.turns_per_day * assignment.day + assignment.start_turn
          y = self.turns_per_day * ideal_day + ideal_turn
          deviations.append(abs(x - y))
        if deviations:
          deviation = sum(deviations) / len(deviations)
        else:
          deviation = 0
        flex_dev += deviation
    
    return flex_dev

  def calculate_robustness(self, solution: Solution) -> float:
    "Calcula el componente de robustez R(x)"
    margin = self.calculate_margin(solution)
    flex_dev = self.calculate_flex_deviation(solution)
    
    return self.delta * margin - self.lambda_param * flex_dev

  def evaluate(self, solution: Solution) -> float:
    "Evalúa una solución y asigna su fitness"
    
    solution.is_penalty = True
    if not self.is_feasible(solution):
      solution.fitness = -float('inf')
      solution.penalty = float('inf')
      return solution.fitness
    
    solution.penalty = self.calculate_penalty(solution)
    
    if solution.penalty >= self.penalty_threshold:
      solution.fitness = -float('inf')
      return solution.fitness
    
    solution.is_penalty = False
    adaptability = self.calculate_adaptability(solution)
    robustness = self.calculate_robustness(solution)
    
    solution.fitness = self.gamma1 * adaptability + self.gamma2 * robustness
    return solution.fitness


@dataclass
class GeneticConfig:    
  population_size:int   # tamaño de la población
  elite_size:int        # tamaño de la élite
  tournament_size:int   # tamaño del torneo
  crossover_rate:float  # probabilidad de cruce
  mutation_rate:float   # probabilidad de mutación
  max_generations:int   # número máximo de generaciones
  repair_attempts:int   # número de iteraciones de búsqueda local

  @staticmethod
  def generate_random_config(fixed_param: str = None, fixed_value = None) -> 'GeneticConfig':
    """Genera una configuración aleatoria para el algoritmo genético
    
    Args:
      fixed_param: nombre del parámetro fijo (ej: 'population_size', 'elite_size_percent')
      fixed_value: valor fijo para el parámetro especificado
    
    Returns:
      GeneticConfig: configuración aleatoria generada
    """
    # Valores posibles para cada parámetro
    population_sizes = [100, 200, 300, 500, 800, 1000]
    elite_percentages = [0.01, 0.05, 0.10, 0.15, 0.20]  # 1%, 5%, 10%, 15%, 20%
    tournament_percentages = [0.10, 0.15, 0.20]  # 10%, 15%, 20%
    crossover_rates = [0.4, 0.6, 0.8, 0.9]
    mutation_rates = [0.4, 0.6, 0.8, 0.9]
    max_generations_options = [200, 300, 400, 500]
    repair_percentages = [0.10, 0.20]  # 10%, 20% del max_generations
    
    # Determinar population_size
    if fixed_param == 'population_size':
      population_size = fixed_value
    else:
      population_size = random.choice(population_sizes)
    
    # Determinar elite_size
    if fixed_param == 'elite_size_percent':
      elite_percentage = fixed_value
    else:
      elite_percentage = random.choice(elite_percentages)
    elite_size = max(1, int(population_size * elite_percentage))
    
    # Determinar tournament_size  
    if fixed_param == 'tournament_size_percent':
      tournament_percentage = fixed_value
    else:
      tournament_percentage = random.choice(tournament_percentages)
    tournament_size = max(2, int(population_size * tournament_percentage))
    
    # Determinar crossover_rate
    if fixed_param == 'crossover_rate':
      crossover_rate = fixed_value
    else:
      crossover_rate = random.choice(crossover_rates)
    
    # Determinar mutation_rate
    if fixed_param == 'mutation_rate':
      mutation_rate = fixed_value
    else:
      mutation_rate = random.choice(mutation_rates)
    
    # Determinar max_generations
    if fixed_param == 'max_generations':
      max_generations = fixed_value
    else:
      max_generations = random.choice(max_generations_options)
    
    # Determinar repair_attempts
    if fixed_param == 'repair_attempts_percent':
      repair_percentage = fixed_value
    else:
      repair_percentage = random.choice(repair_percentages)
    repair_attempts = max(1, int(max_generations * repair_percentage))
    
    return GeneticConfig(
      population_size=population_size,
      elite_size=elite_size,
      tournament_size=tournament_size,
      crossover_rate=crossover_rate,
      mutation_rate=mutation_rate,
      max_generations=max_generations,
      repair_attempts=repair_attempts
    )
  
  @staticmethod
  def generate_multiple_configs(n_configs: int, fixed_param: str = None, 
                                fixed_value = None) -> List['GeneticConfig']:
    """Genera múltiples configuraciones aleatorias
    
    Args:
      n_configs: número de configuraciones a generar
      fixed_param: parámetro fijo (opcional)
      fixed_value: valor del parámetro fijo (opcional)
    
    Returns:
      List[GeneticConfig]: lista de configuraciones generadas
    """
    configs = []
    for _ in range(n_configs):
      config = GeneticConfig.generate_random_config(fixed_param, fixed_value)
      configs.append(config)
    return configs
  
class GeneticAlgorithm:
  "Implementación del algoritmo genético para el problema de asignación"
  
  def __init__(self, problem:TaskSchedulerProblem, config:GeneticConfig):
    self.problem = problem
    self.population_size = config.population_size
    self.elite_size = config.elite_size
    self.tournament_size = config.tournament_size
    self.crossover_rate = config.crossover_rate
    self.mutation_rate = config.mutation_rate
    self.max_generations = config.max_generations
    self.repair_attempts = config.repair_attempts
    
  def generate_initial_population(self) -> List[Solution]:
    "Genera la población inicial"
    population = []
    
    for _ in range(self.population_size):
      assignments = []
      occupied_slots = set()
      
      for task in self.problem.tasks:
        assignment = self._generate_valid_assignment(task, occupied_slots)
        if assignment:
          assignments.append(assignment)
          # Marcar turnos ocupados 
          for turn in range(assignment.start_turn, assignment.start_turn + task.duration):
            occupied_slots.add((assignment.day, turn))
      
      solution = Solution(assignments)
      self._repair_solution(solution, self.repair_attempts)
      population.append(solution)
    
    return population

  def _generate_valid_assignment(self, task: Task, occupied_slots: Set[Tuple[int, int]]) -> Optional[Assignment]:
    "Genera una asignación válida para una tarea"
    max_attempts = 100
    
    for _ in range(max_attempts):
      # Seleccionar día válido
      if task.valid_days:
        day = random.choice(list(task.valid_days))
      else:
        day = random.choice([d.id for d in self.problem.days])
      
      # Seleccionar turno válido
      day_obj = self.problem.day_map[day]
      available_turns = []
      
      for turn in task.valid_turns:
        # Verificar que la tarea complete antes del final del día
        if turn + task.duration <= self.problem.turns_per_day:
          # Verificar que no haya eventos fijos
          conflict = False
          for s in range(turn, turn + task.duration):
            if s in day_obj.busy_slots or (day, s) in occupied_slots:
              conflict = True
              break
          
          if not conflict:
            available_turns.append(turn)
      
      if available_turns:
        turn = random.choice(available_turns)
        return Assignment(task.id, day, turn)
    
    # Si no se puede asignar respetando restricciones, asignar aleatoriamente
    day = random.choice([d.id for d in self.problem.days])
    turn = random.choice(list(task.valid_turns) if task.valid_turns else list(range(self.problem.turns_per_day)))
    turn = max(0, min(turn, self.problem.turns_per_day - task.duration))
    
    return Assignment(task.id, day, turn)

  def _repair_solution(self, solution: Solution, attempts: int):
    "Intenta reparar una solución para mejorar las restricciones blandas"
    for _ in range(attempts):
      if solution.penalty < self.problem.penalty_threshold:
        break
      
      # Seleccionar tarea aleatoria para reparar
      task_id = random.choice([t.id for t in self.problem.tasks])
      assignment = solution.get_assignment(task_id)
      task = self.problem.task_map[task_id]
      
      # Intentar mejorar la asignación
      best_penalty = self.problem.calculate_penalty(solution)
      best_assignment = assignment
      
      # Probar diferentes días y turnos válidos
      for day in task.valid_days:
        for turn in task.valid_turns:
          temp_assignment = Assignment(task_id, day, turn)
          temp_solution = solution.copy()
          temp_solution.assignment_map[task_id] = temp_assignment
          
          if self.problem.is_feasible(temp_solution):
            penalty = self.problem.calculate_penalty(temp_solution)
            if penalty < best_penalty:
              best_penalty = penalty
              best_assignment = temp_assignment
      
      # Actualizar si se encontró una mejor asignación
      if best_assignment != assignment:
        solution.assignment_map[task_id] = best_assignment
        for i, a in enumerate(solution.assignments):
          if a.task_id == task_id:
            solution.assignments[i] = best_assignment
            break

  def tournament_selection(self, population: List[Solution]) -> Solution:
    "Selección por torneo binario"
    tournament = random.sample(population, min(self.tournament_size, len(population)))
    return max(tournament, key=lambda s: s.fitness)

  def block_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
    "Cruce basado en bloques"
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Identificar tareas que comparten días o turnos permitidos
    shared_tasks = []
    for task in self.problem.tasks:
      p1_assignment = parent1.get_assignment(task.id)
      p2_assignment = parent2.get_assignment(task.id)
      
      if (p1_assignment.day in task.valid_days and p2_assignment.day in task.valid_days):
        shared_tasks.append(task.id)
    
    # Intercambiar asignaciones de tareas compartidas
    if shared_tasks:
      tasks_to_swap = random.sample(shared_tasks, max(1, len(shared_tasks) // 2))
      
      for task_id in tasks_to_swap:
        p1_assignment = parent1.get_assignment(task_id)
        p2_assignment = parent2.get_assignment(task_id)
        
        child1.assignment_map[task_id] = p2_assignment
        child2.assignment_map[task_id] = p1_assignment
      
      # Actualizar listas de asignaciones
      child1.assignments = list(child1.assignment_map.values())
      child2.assignments = list(child2.assignment_map.values())
    
    return child1, child2

  def uniform_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
    "Cruce uniforme"
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for task in self.problem.tasks:
      if random.random() < 0.5:
        # Intercambiar asignaciones
        p1_assignment = parent1.get_assignment(task.id)
        p2_assignment = parent2.get_assignment(task.id)
        
        child1.assignment_map[task.id] = p2_assignment
        child2.assignment_map[task.id] = p1_assignment
    
    # Actualizar listas de asignaciones
    child1.assignments = list(child1.assignment_map.values())
    child2.assignments = list(child2.assignment_map.values())
    
    return child1, child2

  def local_reassignment_mutation(self, solution: Solution):
    "Mutación por reasignación local"
    if random.random() > self.mutation_rate:
      return
    
    task_id = random.choice([t.id for t in self.problem.tasks])
    assignment = solution.get_assignment(task_id)
    task = self.problem.task_map[task_id]
    
    # Intentar reasignar dentro de la ventana de tiempo preferida
    current_day, current_turn = assignment.day, assignment.start_turn
    
    if (current_day in task.valid_days and current_turn in task.valid_turns):
      # Buscar alternativas dentro de la ventana
      alternatives = []
      for day in task.valid_days:
        for turn in task.valid_turns:
          if (day, turn) != (current_day, current_turn):
            temp_assignment = Assignment(task_id, day, turn)
            temp_solution = solution.copy()
            temp_solution.assignment_map[task_id] = temp_assignment
            
            if self.problem.is_feasible(temp_solution):
              alternatives.append((day, turn))
      
      if alternatives:
        new_day, new_turn = random.choice(alternatives)
        new_assignment = Assignment(task_id, new_day, new_turn)
        solution.assignment_map[task_id] = new_assignment
        
        # Actualizar lista
        for i, a in enumerate(solution.assignments):
          if a.task_id == task_id:
            solution.assignments[i] = new_assignment
            break

  def swap_mutation(self, solution: Solution):
    "Mutación por intercambio"
    if random.random() > self.mutation_rate:
      return
    
    if len(self.problem.tasks) < 2:
      return
    
    # Seleccionar dos tareas aleatoriamente
    task_ids = random.sample([t.id for t in self.problem.tasks], 2)
    
    assignment1 = solution.get_assignment(task_ids[0])
    assignment2 = solution.get_assignment(task_ids[1])
    
    # Crear asignaciones intercambiadas
    swapped1 = Assignment(task_ids[0], assignment2.day, assignment2.start_turn)
    swapped2 = Assignment(task_ids[1], assignment1.day, assignment1.start_turn)
    
    # Verificar factibilidad del intercambio
    temp_solution = solution.copy()
    temp_solution.assignment_map[task_ids[0]] = swapped1
    temp_solution.assignment_map[task_ids[1]] = swapped2
    
    if self.problem.is_feasible(temp_solution):
      solution.assignment_map[task_ids[0]] = swapped1
      solution.assignment_map[task_ids[1]] = swapped2
      
      # Actualizar lista
      for i, a in enumerate(solution.assignments):
        if a.task_id == task_ids[0]:
          solution.assignments[i] = swapped1
        elif a.task_id == task_ids[1]:
          solution.assignments[i] = swapped2

  def local_search(self, solution: Solution):
    "Búsqueda local iterada"
    improved = True
    max_iterations = 10
    iteration = 0
    
    while improved and iteration < max_iterations:
      improved = False
      iteration += 1
      current_fitness = solution.fitness
      
      # Intentar mejorar cada tarea
      for task in self.problem.tasks:
        assignment = solution.get_assignment(task.id)
        best_fitness = current_fitness
        best_assignment = assignment
        
        # Explorar vecindario (días y turnos adyacentes)
        neighbors = []
        
        # Días adyacentes
        for day_offset in [-1, 0, 1]:
          new_day = assignment.day + day_offset
          if (new_day >= 0 and new_day < len(self.problem.days) and
              new_day in task.valid_days):
            neighbors.append((new_day, assignment.start_turn))
        
        # Turnos adyacentes
        for shift_offset in [-1, 0, 1]:
          new_turn = assignment.start_turn + shift_offset
          if (new_turn >= 0 and 
              new_turn + task.duration <= self.problem.turns_per_day and
              new_turn in task.valid_turns):
            neighbors.append((assignment.day, new_turn))
        
        # Evaluar vecinos
        for day, turn in neighbors:
          if (day, turn) == (assignment.day, assignment.start_turn):
            continue
          
          temp_assignment = Assignment(task.id, day, turn)
          temp_solution = solution.copy()
          temp_solution.assignment_map[task.id] = temp_assignment
          
          if self.problem.is_feasible(temp_solution):
            fitness = self.problem.evaluate(temp_solution)
            if fitness > best_fitness:
              best_fitness = fitness
              best_assignment = temp_assignment
              improved = True
        
        # Aplicar mejor movimiento
        if best_assignment != assignment:
          solution.assignment_map[task.id] = best_assignment
          for i, a in enumerate(solution.assignments):
            if a.task_id == task.id:
              solution.assignments[i] = best_assignment
              break
          solution.fitness = best_fitness

  def evolve(self) -> Solution:
    "Ejecuta el algoritmo genético"
    # Inicializar población
    population = self.generate_initial_population()
    
    # Evaluar población inicial
    for solution in population:
      self.problem.evaluate(solution)
    
    # Encontrar mejor solución inicial
    best_solution = max(population, key=lambda s: s.fitness)
    
    print(f"Fitness inicial: {best_solution.fitness}")
    print(f"Number of non-penalty-solutions: {sum([1 for s in population if s.is_penalty])}")
    
    # Evolución
    for generation in range(self.max_generations):
      new_population = []
      
      # Preservar élites
      population.sort(key=lambda s: s.fitness, reverse=True)
      elites = population[:self.elite_size]
      new_population.extend([elite.copy() for elite in elites])
      
      # Generar descendencia
      while len(new_population) < self.population_size:
        # Selección
        parent1 = self.tournament_selection(population)
        parent2 = self.tournament_selection(population)
        
        # Cruce
        if random.random() < self.crossover_rate:
          if random.random() < 0.5:
            child1, child2 = self.block_crossover(parent1, parent2)
          else:
            child1, child2 = self.uniform_crossover(parent1, parent2)
        else:
          child1, child2 = parent1.copy(), parent2.copy()
        
        # Mutación
        self.local_reassignment_mutation(child1)
        self.swap_mutation(child1)
        self.local_reassignment_mutation(child2)
        self.swap_mutation(child2)
        
        # Reparar soluciones
        self._repair_solution(child1, 2)
        self._repair_solution(child2, 2)
        
        new_population.extend([child1, child2])
      
      # Limitar tamaño de población
      new_population = new_population[:self.population_size]
      
      # Evaluar nueva población
      for solution in new_population:
        self.problem.evaluate(solution)
      
      # Aplicar búsqueda local a los mejores
      best_solutions = sorted(new_population, key=lambda s: s.fitness, reverse=True)[:5] # fix
      for solution in best_solutions:
        if not solution.is_penalty:
          self.local_search(solution)
      
      # Actualizar población
      population = new_population
      
      # Actualizar mejor solución
      non_penalty_solutions = []
      for solution in population:
        if not solution.is_penalty:
          non_penalty_solutions.append(solution)
      
      if len(non_penalty_solutions) > 0:
        current_best = max([solution for solution in non_penalty_solutions], key=lambda solution: solution.fitness)
        if current_best.fitness > best_solution.fitness:
          best_solution = current_best.copy()
        if generation % 50 == 0:
          avg_fitness = sum(solution.fitness for solution in non_penalty_solutions) / len(non_penalty_solutions)
          print(f"Promedio: {avg_fitness}\nMejor Solución: {current_best.fitness}")
      
    return best_solution

def generate_task_description(task_id:int) -> str:
  "Genera una descripción aleatoria para la tarea"
  task_types = [
    "Análisis Matemático",
    "Algebra",
    "Programación",
    "Teoría de Grafos",
    "Estructura de Datos",
    "Inteligencia Artificial",
    "Machine Learning",
    "Ingeniería de Software"
  ]
  return f"{random.choice(task_types)} #={task_id}"

def generate_tasks(k:int, n:int, m:int) -> List[Task]:
  "Genera K tareas con propiedades aleatorias"
  tasks = []
  
  duration = 1
  for i in range(1, k+1):
    # prioridad entre 0.1 y 1.0
    priority = round(random.uniform(0.1, 1.0), 2)

    # carga entre 0.2 y 1.0
    load = round(random.uniform(0.2, 1.0), 2)
    
    # Flexibilidad entre 0.0 y 1.0
    flexibility = round(random.uniform(0.0, 1.0), 2)
    
    # Turnos válidos: al menos duration turnos consecutivos posibles
    num_valid_turns = random.randint(duration, m)
    valid_turns = set(random.sample(range(1, m + 1), num_valid_turns))
    
    # Días válidos: entre 1 y n días
    num_valid_days = random.randint(1, n)
    valid_days = set(random.sample(range(1, n + 1), num_valid_days))
    
    # Turnos ideales: subconjunto de turnos válidos y días válidos
    ideal_turns = None
    if random.random() > 0.3:  # 70% probabilidad de tener turnos ideales
      num_ideal = random.randint(1, min(3, len(valid_days)))
      selected_days = random.sample(list(valid_days), num_ideal)
      ideal_turns = set()
      for day in selected_days:
        available_turns = list(valid_turns)
        selected_turn = random.choice(available_turns)
        ideal_turns.add((day, selected_turn))
    
    task = Task(
      id=i,
      description=generate_task_description(i),
      duration=duration,
      priority=priority,
      load=load,
      flexibility=flexibility,
      valid_turns=valid_turns,
      valid_days=valid_days,
      ideal_turns=ideal_turns
    )
    
    tasks.append(task)
  
  return tasks

def generate_days (n:int, m:int) -> List[Day]:
  "Genera N días con propiedades aleatorias"
  days = []
  
  for i in range(1, n+1):
    # Energía entre 5.0 y 10.0
    energy = round(random.uniform(2.0, 5.0), 2)
    
    # Turnos ocupados: 0 a 30% de los turnos del día
    max_busy = max(1, int(m * 0.3))
    num_busy = random.randint(0, max_busy)
    busy_slots = set(random.sample(range(1, m + 1), num_busy))
    
    day = Day(
      id=i,
      energy=energy,
      busy_slots=busy_slots
    )
    
    days.append(day)
  
  return days

def convert_to_serializable(obj):
  "Convierte sets a listas para serialización JSON"
  if isinstance(obj, set):
    return list(obj)
  elif isinstance(obj, dict):
    return {key: convert_to_serializable(value) for key, value in obj.items()}
  elif isinstance(obj, list):
    return [convert_to_serializable(item) for item in obj]
  elif isinstance(obj, tuple):
    return list(obj)
  else:
    return obj

def generate_data(n: int, m: int, k: int, filename: str = "generated_data.json") -> dict:
  """
  Genera datos de tareas y días y los guarda en JSON
  
  Args:
    n: cantidad de días
    m: cantidad de turnos por día
    k: cantidad de tareas
    filename: nombre del archivo JSON de salida
    
  Returns:
    dict: diccionario con los datos generados
  """
  
  # Validación de parámetros
  if n <= 0 or m <= 0 or k <= 0:
    raise ValueError("N, M y K deben ser valores positivos")
  
  # Generar tareas y días
  tasks = generate_tasks(k, n, m)
  days = generate_days(n, m)
  
  # Crear estructura de datos
  data = {
    "parameters": {
      "n_days": n,
      "m_turns": m,
      "k_tasks": k
    },
    "tasks": [asdict(task) for task in tasks],
    "days": [asdict(day) for day in days]
  }
  
  # Convertir sets a listas para JSON
  data = convert_to_serializable(data)
  
  # Guardar en JSON con identación 2
  with open(filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
  
  print(f"Datos generados y guardados en {filename}")
  print(f"- {k} tareas generadas")
  print(f"- {n} días generados")
  print(f"- {m} turnos por día")
  
  return data


def load_data_from_json(filename: str) -> Tuple[List[Task], List[Day], int]:
  """
  Carga datos desde archivo JSON y devuelve tareas, días y turnos por día
  
  Args:
    filename: nombre del archivo JSON a cargar
    
  Returns:
    Tuple[List[Task], List[Day], int]: (tasks, days, turns_per_day)
  """
  try:
    with open(filename, 'r', encoding='utf-8') as f:
      data = json.load(f)
    
    # Extraer parámetros
    turns_per_day = data['parameters']['m_turns']
    
    # Reconstruir tareas
    tasks = []
    for task_data in data['tasks']:
      # Convertir listas de vuelta a sets
      task_data['valid_turns'] = set(task_data['valid_turns'])
      task_data['valid_days'] = set(task_data['valid_days'])
      
      # Manejar ideal_turns (puede ser None o lista de tuplas)
      if task_data['ideal_turns'] is not None:
        task_data['ideal_turns'] = set(
          tuple(item) if isinstance(item, list) else item 
          for item in task_data['ideal_turns']
        )
      
      task = Task(**task_data)
      tasks.append(task)
    
    # Reconstruir días
    days = []
    for day_data in data['days']:
      # Convertir lista de vuelta a set
      day_data['busy_slots'] = set(day_data['busy_slots'])
      
      day = Day(**day_data)
      days.append(day)
    
    print(f"Datos cargados desde {filename}")
    print(f"- {len(tasks)} tareas cargadas")
    print(f"- {len(days)} días cargados")
    print(f"- {turns_per_day} turnos por día")
    
    return tasks, days, turns_per_day
    
  except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo {filename}")
  except KeyError as e:
    raise ValueError(f"Formato de archivo inválido: falta la clave {e}")
  except Exception as e:
    raise ValueError(f"Error al cargar los datos: {str(e)}")


def test() -> None:
  data = generate_data(n=7, m=10, k=20, filename="generated_data.json")
  tasks, days, turns_per_day = load_data_from_json("generated_data.json")
  
  paramConfig = ParamsConfig(
    gamma1=1.0,
    gamma2=1.0,
    delta=1.0,
    lambda_param=1.0,
    penalty_threshold=2.0,
    beta_window=0.5,
    beta_valid_days=0.5
  )
  problem = TaskSchedulerProblem(tasks, days, turns_per_day, paramConfig) 
  geneticConfig = GeneticConfig(
    population_size=100,
    elite_size=20,
    tournament_size=5,
    crossover_rate=0.6,
    mutation_rate=0.6,
    max_generations=200,
    repair_attempts=5
  )
  ga = GeneticAlgorithm(problem=problem, config=geneticConfig)
  best_solution = ga.evolve()
  print(f"Mejor Solución\n{best_solution}")

def run_scheduling() -> Solution:
  tasks, days, turns_per_day = load_data_from_json("generated_data.json")
  
  paramConfig = ParamsConfig(
    gamma1=1.0,
    gamma2=1.0,
    delta=1.0,
    lambda_param=1.0,
    penalty_threshold=2.0,
    beta_window=0.5,
    beta_valid_days=0.5
  )
  problem = TaskSchedulerProblem(tasks, days, turns_per_day, paramConfig) 
  geneticConfig = GeneticConfig(
    population_size=100,
    elite_size=20,
    tournament_size=5,
    crossover_rate=0.6,
    mutation_rate=0.6,
    max_generations=200,
    repair_attempts=5
  )
  ga = GeneticAlgorithm(problem=problem, config=geneticConfig)
  best_solution = ga.evolve()
  print(f"Mejor Solución\n{best_solution}")
  return best_solution

if __name__ == "__main__":
  test()