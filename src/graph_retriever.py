import json 
from typing import List, Dict, Optional, Tuple, Set 
from collections import deque, defaultdict
from pymongo import MongoClient

from src.graph_db import GraphDB 


def find_paths_between_entities(self:GraphDB,
      source:str,
      target:str,
      max_depth:int,
      max_paths:int ) -> Dict:
  """Encuentra todos los caminos posibles entre dos entidades usando BFS con múltiples caminos 

  Args:
      source (str): entidad origen
      target (str): entidad destino
      max_depth (int): profundidad máxima de búsqueda
      max_paths (int): número máximo de caminos a retornar

  Returns:
      Dict: información sobre los caminos encontrados
  """
  # validar que ambas entidades existen
  if not self._node_exists(target):
    return {
      "success": False,
      "message": f"La entidad '{target}' no existe en el grafo",
      "query_type": "path_finding"
    }
  
  # verificar si ya existe una relación directa 
  direct_relations = self.get_adjacent_entities(source)
  direct_target = next( (rel for rel in direct_relations if rel["node_id"] == target), None )
  
  if direct_target:
    return {
      "success": True,
      "query_type": "direct_relation",
      "source": source,
      "target": target,
      "direct_relation": direct_target,
      "message": f"Existe una relación directa entre `{source}` y `target`"
    }
  
  # BFS para encontrar múltiples caminos
  paths_found = []
  queue = deque([(source, [source], 0)])  # nodo_actual, camino, profundidad
  visited_paths = set()                   # para evitar caminos duplicados
  
  while queue and len(paths_found) < max_paths:
    current_node, path, depth = queue.popleft()
    if depth >= max_depth: continue
    
    # obtener entidades adyacentes
    adjacent = self.get_adjacent_entities(current_node)
    
    for adj_entity in adjacent:
      next_node = adj_entity["node_id"]
      
      # si encontramos el objetivo
      if next_node == target:
        complete_path = path + [next_node]
        path_key = "->".join(complete_path)
        
        if path_key not in visited_paths:
          visited_paths.add(path_key)

          # construir información detallada del camino
          path_details = self._build_path_details(complete_path)
          
          paths_found.append({
            "path": complete_path,
            "length": len(complete_path) - 1,
            "details": path_details
          })
      
      # si no se ha visitado este nodo en este camino, continuar explorando
      elif next_node not in path:
        new_path = path + [next_node]
        queue.append((next_node,new_path,depth + 1))
    
  # ordenar caminos por longitud 
  paths_found.sort(key=lambda x:x["length"])
  
  return {
    "success": True,
    "query_type": "path_finding",
    "source": source,
    "target": target,
    "paths_found": len(paths_found),
    "paths": paths_found[:max_paths],
    "message": f"Se encontraron {len(paths_found)} caminos entre `{source}` y `{target}`"
  }

def calculate_relation_probabilities(self:GraphDB,
      entity:str,
      relation_type:str,
      max_depth:int ) -> Dict:
  """Calcula la probabilidad de que una entidad tenga una relación específica con otras entidades basándose en caminos indirectos y patrones en el grafo

  Args:
      entity (str): entidad base
      relation_type (str): tipo de relación a analizar
      max_depth (int): profundidad máxima para análisis

  Returns:
      Dict: Probabilidades calculadas para cada entidad
  """
  
  # verificar si el tipo de relación existe en el grafo
  existing_relations = self.get_relationship_types()
  if relation_type not in existing_relations:
    return {
      "success": False,
      "message": f"El tipo de relación `{relation_type}` no existe en el grafo",
      "available_relations": existing_relations,
      "query_type": "relation_probability"
    }
  
  # obtener todas las relaciones directas de la entidad 
  direct_relations = self.get_adjacent_entities(entity)
  direct_relation_targets = {rel["node_id"] for rel in direct_relations if rel["relationship"]["type"] == relation_type}
  
  # obtener todos los nodos del grafo
  all_nodes = list(self.nodes.find({}, {"node_id": 1, "type": 1}))
  
  # calcular probabilidades para cada nodo
  probabilities = []
  
  for node_doc in all_nodes:
    target_node = node_doc["node_id"]
    if target_node == entity: continue
    
    # si ya existe una relación directa, entonces probabilidad = 1.0
    if target_node in direct_relation_targets:
      probabilities.append({
        "node_id": target_node,
        "node_type": node_doc["type"],
        "probability": 1.0,
        "message": "Relación directa encontrada",
        "evidence": []
      })
      continue
    
    # calcular probabilidad basada en algunos factores 
    probability, evidence = self._calculate_indirect_probability(entity, target_node, relation_type, max_depth)
    if probability > 0:
      probabilities.append({
        "node_id": target_node,
        "node_type": node_doc["type"],
        "probability": probability,
        "message": "Relación indirecta encontrada basada en patrones indirectos",
        "evidence": evidence
      })
  
  # ordenar por probabilidad (mayor primero)
  probabilities.sort(key=lambda x:x["probability"], reverse=True)
  
  return {
    "success": True,
    "query_type": "relation_probability",
    "entity": entity,
    "relation_type": relation_type,
    "total_candidates": len(probabilities),
    "direct_relations": len(direct_relation_targets),
    "probabilities": probabilities,  
    "message": f"Calculadas probabilidades de relación '{relation_type}' para '{entity}'"
  }

def calculate_indirect_probability(self:GraphDB,
      source:str,
      target:str,
      relation_type:str,
      max_depth:int ) -> Tuple[float, List[Dict]]:
  """Calcula la probabilidad indirecta de relación basada en:
  1. Entidades comunes conectadas con el mismo tipo de relación
  2. Caminos cortos entre las entidades

  Args:
      source (str): entidad origen
      target (str): entidad objetivo
      relation_type (str): tipo de relación
      max_depth (int): profundidad máxima

  Returns:
      Tuple[float, List[Dict]]: probabilidad y evidencia
  """
  evidence = []
  probability_factors = []
  
  # factor 1: entidades comunes conectadas con el mismo tipo de relación
  source_relations = self.get_adjacent_entities(source, relation_filter=relation_type)
  target_relations = self.get_adjacent_entities(target, relation_filter=relation_type)
  
  source_connected = {rel["node_id"] for rel in source_relations}
  target_connected = {rel["node_id"] for rel in target_relations}
  
  common_entities = source_connected.intersection(target_connected)
  
  if common_entities:
    # m: entidades conectadas a source
    m = len(source_connected)
    # n: entidades conectadas a target
    n = len(target_connected)
    # p: entidades comunes entre source y target
    p = len(common_entities)
    
    # máximo de conexiones comunes posibles 
    common_max = min(m,n)
    
    common_factor = 0.0 # evitar división por cero
    if common_max != 0: common_factor = min(p/common_max, 1.0) # P(A) = conexiones-reales / máximo-posible
    
    probability_factors.append(common_factor)
    evidence.append({
      "type": "common_entities",
      "count": len(common_entities),
      "entities": list(common_entities),
      "contribution": common_factor
    })
  
  # factor 2: caminos cortos entre entidades => P(B) = 1/d(i,j)
  paths_result = self._find_paths_between_entities(source,target,min(max_depth, 3), 5)
  if paths_result["success"] and "paths" in paths_result:
    shortest_path_length = min(path["length"] for path in paths_result["paths"]) if paths_result["paths"] else float('inf')
    if shortest_path_length != float('inf'): 
      path_factor = 1/shortest_path_length
      
      probability_factors.append(path_factor)
      evidence.append({
        "type": "shortest_path", 
        "shortest_length": shortest_path_length,
        "total_paths": len(paths_result["paths"]),
        "contribution": path_factor
      })
  
  # unión de probabilidades independientes: P(A u B) = P(A) + P(B) - P(A)*P(B)
  if probability_factors:
    combined_probability = probability_factors[0]
    for factor in probability_factors[1:]:
      combined_probability = combined_probability + factor - (combined_probability * factor)
    return combined_probability, evidence
  
  return 0.0, evidence

def build_path_details(self:GraphDB, path:List[str]) -> List[Dict]:
  """Construye información detallada para un camino específico

  Args:
      path (List[str]): lista de nodos en el camino

  Returns:
      List[Dict]: detalles de cada paso en el camino
  """
  details = []
  for i in range(len(path) - 1):
    current_node = path[i]
    next_node = path[i+1]

    # encontrar la relación entre estos dos nodos
    current_adjacent = self.get_adjacent_entities(current_node)
    relation_info = next((rel for rel in current_adjacent if rel["node_id"] == next_node), None)

    if relation_info:
      details.append({
        "from": current_node,
        "to": next_node,
        "relation_type": relation_info["relationship"]["type"],
        "direction": relation_info["relationship"]["direction"],
        "properties": relation_info["relationship"]["properties"]
      })
  
  return details

def retriever(self:GraphDB, 
      query_entity_1:str, 
      query_entity_2:str = None,
      relation_type:str = None,
      max_depth:int = 4, 
      max_paths:int = 10 ) -> Dict:
  """Recuperador avanzado para el grafo que maneja dos casos principales:
  1. Encontrar caminos entre dos entidades no adyacentes
  2. Calcular probabilidades de la relación entre una entidad y todas las demás para un tipo de relación específico

  Args:
      query_entity_1 (str): primera entidad (siempre requerida)
      query_entity_2 (str, optional): segunda entidad para encontrar caminos. Defaults to None.
      relation_type (str, optional): tipo de relación para calcular probabilidades. Defaults to None.
      max_depth (int, optional): profundidad máxima de búsqueda. Defaults to 4.
      max_paths (int, optional): número máximo de caminos a retornar. Defaults to 10.

  Returns:
      Dict: Resultado con caminos encontrados o probabilidades calculadas
  """
  # validar que la primera entidad existe
  if not self._node_exists(query_entity_1):
    return {
      "success": False,
      "message": f"La entidad '{query_entity_1}' no existe en el grafo",
      "query_type": "error"
    }
  
  # caso 1: encontrar caminos entre dos entidades específicas
  if query_entity_2 is not None:
    return self._find_paths_between_entities(query_entity_1,query_entity_2,max_depth,max_paths)
  
  # caso 2: calcular probabilidades de relación para un tipo específico 
  elif relation_type is not None:
    return self._calculate_relation_probabilities(query_entity_1, relation_type, max_depth)
  
  # error: parámetros insuficientes
  else:
    return {
      "success": False,
      "message": "Debe proporcionar `query_entity_2` (para encontrar caminos) o `relation_type` (para calcular probabilidades)",
      "query_type": "error"
    }

