from random import choice
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from typing import List, Dict, Optional, Tuple
from pymongo import MongoClient
from collections import deque


class GraphDB:
  def __init__(self, db_name:str = "graph_db", host:str = "localhost", port:int = 27017) -> None:
    """Inicializa la conexión a MongoDB y crea las colecciones necesarias

    Args:
        db_name (str, optional): nombre de la base de datos. Defaults to "graph_db".
        host (str, optional): host de mongodb. Defaults to "localhost".
        port (int, optional): puerto de mongodb. Defaults to 27017.
    """
    
    self.client:MongoClient = MongoClient(host,port)
    self.db = self.client[db_name]
    
    # colecciones
    self.nodes = self.db.nodes # entidades/nodos del grafo
    self.edges = self.db.edges # relaciones/aristas del grafo
    
    # crear indices para mejorar el rendimiento
    self._create_indexes()
  
  def _create_indexes(self):
    "Crea índices para optimizar las consultas"
    # índices para nodos
    self.nodes.create_index("node_id", unique=True)
    self.nodes.create_index("type")
    
    # índices para aristas
    self.edges.create_index([("source", 1), ("target", 1)])
    self.edges.create_index("source")
    self.edges.create_index("target")
    self.edges.create_index("relation_type")
  
  def add_node(self, node_id:str, node_type:str, properties:Dict = None) -> bool:
    """Añade un nodo al grafo
  
    Args:
        node_id (str): identificador único del nodo
        node_type (str): tipo de nodo
        properties (Dict, optional): propiedades adicionales del nodo. Defaults to None.

    Returns:
        bool: True si se añadió correctamente, False en caso contrario
    """
    if properties is None: properties = { }
    
    node_doc = {
      "node_id": node_id,
      "type": node_type,
      "properties": properties
    }
    
    try:
      self.nodes.insert_one(node_doc)
      return True
    except Exception as e:
      print(f"Error al insertar nodo: {e}")
      return False
  
  def del_node(self, node_id:str, cascade: bool = True) -> Dict:
    """Elimina un nodo del grafo

    Args:
        node_id (str): ID del nodo a eliminar
        cascade (bool, optional): Si True, elimina también todas las relaciones del nodo. Si False, solo elimina el nodo si no tiene relaciones. Defaults to True.

    Returns:
        Dict: Resultado de la operación de eliminación
    """
    if not self._node_exists(node_id=node_id):
      return {
        "success": False,
        "message": f"El nodo {node_id} no existe",
        "nodes_deleted": 0,
        "edges_deleted": 0,
        "existing_edges": 0,
      }
    
    # contar relaciones existentes
    outgoing_edges = self.edges.count_documents({"source": node_id})
    incoming_edges = self.edges.count_documents({"target": node_id})
    total_edges = outgoing_edges + incoming_edges
    
    # si cascade es False y tiene relaciones, no eliminar
    if not cascade and total_edges > 0:
      return {
        "success": False,
        "message": f"El nodo {node_id} tiene {total_edges} relaciones. Para eliminar el nodo con las conexiones se debe usar cascade=True",
        "nodes_deleted": 0,
        "edges_deleted": 0,
        "existing_edges": total_edges
      }
    
    try:
      # eliminar todas las relaciones donde el nodo es origen o destino
      edges_deleted = 0
      if cascade:
        # eliminar aristas donde es origen (source)
        result_outgoing = self.edges.delete_many({"source":node_id})
        # eliminar aristas donde es destino (target)
        result_incoming = self.edges.delete_many({"target":node_id})
        edges_deleted = result_outgoing.deleted_count + result_incoming.deleted_count
      
      # eliminar el nodo
      result_node = self.nodes.delete_one({"node_id": node_id})
      nodes_deleted = result_node.deleted_count
      
      return {
        "success": True,
        "message": f"Node {node_id} eliminado con exito",
        "nodes_deleted": nodes_deleted,
        "edges_deleted": edges_deleted,
        "existing_edges": 0
      }
    except Exception as e:
      return {
        "success": False,
        "message": f"Error al eliminar nodo: {str(e)}",
        "nodes_deleted": 0,
        "edges_deleted": 0,
        "existing_edges": 0
      }
  
  def add_edge(self, source_id:str, target_id:str, relation_type:str,
              properties: Dict = None, directed: bool = True) -> Dict:
    """Añade una arista (relación) entre dos nodos. Previene la creación de aristas exactamente iguales

    Args:
        source_id (str): ID del nodo origen
        target_id (str): ID del nodo destino
        relation_type (str): Tipo de relación (ej: 'knows', 'works_at', etc)
        properties (Dict, optional): Propiedades adicionales de la relación. Defaults to None.
        directed (bool, optional): Si la relación es dirigida o no. Defaults to True.

    Returns:
        Dict: Resultado de la operación
    """
    if properties is None:
      properties = { }
    
    # verificar que ambos nodos existen
    if not self._node_exists(source_id) or not self._node_exists(target_id):
      return {
        "success": False,
        "message": "Error: Uno o ambos nodos no existen",
        "edge_created": False,
        "duplicate_found": False
      }
    
    # verificar si la arista ya existe
    existing_edge = self._edge_exists(source_id, target_id, relation_type, properties, directed)
    if existing_edge:
      return {
        "success": False,
        "message": f"La arista ya existe: {source_id} -> {target_id} ({relation_type})",
        "edge_created": False,
        "duplicate_found": True,
        "existing_edge_id": str(existing_edge["_id"])
      }
    
    edge_doc = {
      "source": source_id,
      "target": target_id,
      "relation_type": relation_type,
      "properties": properties,
      "directed": directed
    }
    
    try:
      # insertar relación source_id -> target_id
      result = self.edges.insert_one(edge_doc)
      edges_created = 1
      
      # si no es dirigido, crear la relación inversa (target_id -> source_id)
      reverse_created = False 
      if not directed:
        # verificar si la relación inversa ya existe
        reverse_existing = self._edge_exists(target_id, source_id, relation_type, properties, directed)
        if not reverse_existing:
          reverse_edge = {
            "source": target_id,
            "target": source_id,
            "relation_type": relation_type,
            "properties": properties,
            "directed": False
          } 
          self.edges.insert_one(reverse_edge)
          edges_created += 1
          reverse_created = True
      
      return {
        "success": True,
        "message": f"Arista creada exitosamente: {source_id} -> {target_id} ({relation_type})",
        "edge_created": True,
        "duplicate_found": False,
        "edge_id": str(result.inserted_id),
        "edges_created": edges_created,
        "reverse_created": reverse_created
      }  
      
    except Exception as e:
      return {
        "success": False,
        "message": f"Error al insertar arista: {str(e)}",
        "edge_created": False,
        "duplicate_found": False
      }
  
  def _node_exists(self, node_id:str) -> bool:
    "Verifica si un nodo existe"
    return self.nodes.find_one({"node_id":node_id}) is not None
  
  def _edge_exists(self, source_id:str, target_id:str, relation_type:str, properties:Dict, directed:bool) -> Optional[Dict]:
    """Verifica si una arista exactamente igual ya existe

    Args:
        source_id (str): ID del nodo origen
        target_id (str): ID del nodo destino
        relation_type (str): tipo de relación
        properties (Dict): propiedades de la relación
        directed (bool): si la relación es dirigida

    Returns:
        Dict: Información de la arista existente o None si no existe
    """
    query = {
      "source": source_id,
      "target": target_id,
      "relation_type": relation_type,
      "properties": properties,
      "directed": directed
    }
    return self.edges.find_one(query)
  
  def get_adjacent_entities(self, node_id:str, relation_filter:str = None, direction:str = "both") -> List[Dict]:
    """Recupera las entidades adyacentes a un nodo dado

    Args:
        node_id (str): ID del nodo del cual se quiere obtener las entidades adyacentes
        relation_filter (str, optional): Filtro opcional por tipo de relación . Defaults to None.
        direction (str, optional): Dirección de las relacciones ('outgoing', 'incoming', defaults='both')

    Returns:
        List[Dict]: Lista de diccionarios con información de entidades adyacentes
    """
    if not self._node_exists(node_id):
      print(f"Error: El nodo {node_id} no existe")
      return []
    
    query_conditions = []
    
    # Relaciones salientes (este nodo es el origen)
    if direction in ["outgoing", "both"]:
      outgoing_query = {"source":node_id}
      if relation_filter: outgoing_query["relation_type"] = relation_filter
      query_conditions.append(outgoing_query)
      
    # Relaciones entrantes (este nodo es el destino)
    if direction in ["incoming", "both"]:
      incoming_query = {"target":node_id}
      if relation_filter: incoming_query["relation_type"] = relation_filter 
      query_conditions.append(incoming_query)
    
    adjacent_entities:List = []
    if query_conditions:
      if len(query_conditions) == 1:
        edges_cursor = self.edges.find(query_conditions[0])
      else:
        edges_cursor = self.edges.find( {"$or": query_conditions} )
      
      for edge in edges_cursor:
        # determinar cuál es la entidad adyacente
        if edge["source"] == node_id:
          adjacent_id = edge["target"]
          relationship_direction = "outgoing"
        else: 
          adjacent_id = edge["source"]
          relationship_direction = "incoming"
        
        # obtener información de la entidad adyacente
        adjacent_node = self.nodes.find_one( {"node_id": adjacent_id} )
        
        if adjacent_node:
          adjacent_info = {
            "node_id": adjacent_node["node_id"],
            "type": adjacent_node["type"],
            "properties": adjacent_node["properties"],
            "relationship": {
              "type": edge["relation_type"],
              "direction": relationship_direction,
              "properties": edge["properties"],
              "directed": edge["directed"]
            }
          }
          adjacent_entities.append(adjacent_info)
    
    return adjacent_entities  

  def get_all_nodes(self) -> List[Dict]:
    """Obtiene todos los nodos del grafo
    
    Returns:
        List[Dict]: Lista de diccionarios con información de todos los nodos
    """
    nodes_cursor = self.nodes.find({})
    return [node for node in nodes_cursor]

  def get_node_info(self, node_id:str) -> Optional[Dict]:
    """Obtiene información completa del nodo
    
    Args:
        node_id (str): ID del nodo

    Returns:
        Optional[Dict]: Información del nodo o None si no existe
    """
    node = self.nodes.find_one({"node_id":node_id})
    if node:
      # remover el _id de MongoDB para una salida más limpia
      del node['_id']
      return node 
    return None
  
  def get_relationship_types(self) -> List[str]:
    "Obtiene todos los tipos de relaciones únicas en el grafo"
    return self.edges.distinct("relation_type")
  
  def get_node_types(self) -> List[str]:
    "Obtiene todos los tipos de nodos únicos en el grafo"
    return self.nodes.distinct("type")
  
  def get_graph_stats(self) -> Dict:
    "Obtiene estadísticas básicas del grafo"
    return {
      "total_nodes": self.nodes.count_documents({}),
      "total_edges": self.edges.count_documents({}),
      "node_types": self.get_node_types(),
      "relationship_types": self.get_relationship_types()
    }
  
  def clear_database(self, confirm:bool = False) -> Dict:
    """Elimina toda la información de la base de datos (todos los nodos y aristas) 
    
    Args:
        confirm (bool, optional): Debe ser True para confirmar la eliminación total. Esto previene eliminaciones accidentales. Defaults to False.

    Returns:
        Dict: Resultado de la operación con las estadísticas de la eliminación
    """
    if not confirm:
      return {
        "success": False,
        "message": "Eliminación cancelada. Use confirm=True para confirmar la eliminación total de la base de datos",
        "nodes_deleted": 0,
        "edges_deleted": 0,
        "warning": "Esta operación eliminará TODOS los datos permanentemente"
      }
    
    try: 
      # Obtener estadísticas antes de eliminar
      initial_stats = self.get_graph_stats()
      
      # Eliminar todas las aristas
      edges_result = self.edges.delete_many({})
      edges_deleted = edges_result.deleted_count
      
      # Eliminar todos los nodos
      nodes_result = self.nodes.delete_many({})
      nodes_deleted = nodes_result.deleted_count
      
      # Verificar que las colecciones estén vacías
      remaining_nodes = self.nodes.count_documents({})
      remaining_edges = self.edges.count_documents({})
      
      return {
        "success": True,
        "message": "Base de datos limpiada con exito",
        "nodes_deleted": nodes_deleted,
        "edges_deleted": edges_deleted,
        "initial_stats": initial_stats,
        "remaining_nodes": remaining_nodes,
        "remaining_edges": remaining_edges,
        "database_empty": remaining_nodes == 0 and remaining_edges == 0
      }
    except Exception as e:
      return {
        "success": False,
        "message": f"Error al limpiar la base de datos: {str(e)}",
        "nodes_deleted": 0,
        "edges_deleted": 0
      }
  
  def reset_database(self, confirm:bool = False) -> Dict:
    """Reinicia completamente la base de datos: elimina todo y recrea los índices

    Args:
        confirm (bool): Debe ser True para confirmar el reinicio total. Defaults to False.

    Returns:
        Dict: Resultado de la operación
    """
    if not confirm:
      return {
        "success": False,
        "message": "Reinicio cancelado. Use confirm=True para confirmar el reinicio total de la base de datos",
        "warning": "Esta operación eliminará TODOS los datos y recreará los índices"
      }
    try:
      # Limpiar todos los datos
      clear_result = self.clear_database(confirm=True)
      
      if not clear_result["success"]:
        return clear_result
      
      # Eliminar todos los índices existentes
      self.nodes.drop_indexes()
      self.edges.drop_indexes()
      
      # Recrear los índices
      self._create_indexes()
      
      return {
        "success": True,
        "message": "Base de datos reiniciada exitosamente",
        "nodes_deleted": clear_result["nodes_deleted"],
        "edges_deleted": clear_result["edges_deleted"],
        "indexes_recreated": True,
        "database_reset": True
      }
    except Exception as e:
      return {
        "success": False,
        "message": f"Error al reiniciar la base de datos: {str(e)}",
        "nodes_deleted": 0,
        "edges_deleted": 0,
        "indexes_recreated": False
      }
  
  def close_connection(self) -> None:
    "Cierra la conexión a MongoDB"
    self.client.close()

  def extract_subgraph_by_relation(self, relation_type:str) -> Dict:
    """Extrae un subgrafo formado por todas las relaciones de un tipo específico 
    y los nodos que participan en dichas relaciones. 

    Args:
      relation_type (str): tipo de relación para filtrar el subgrafo

    Returns:
      Dict: Diccionario con la información del subgrafo que contiene:
      - `success`: bool indicando si la operación fue exitosa
      - `message`: str con mensaje descriptivo
      - `nodes`: List[Dict] con información de los nodos del subgrafo
      - `edges`: List[Dict] con información de las aristas del subgrafo
      - `stats`: Dict con estadísticas del subgrafo
    """
    try:
      # buscar todas las aristas del tipo de relación especificado
      edges_cursor = self.edges.find({"relation_type": relation_type})
      subgraph_edges = []
      involved_node_ids = set()
      
      # procesar cada arista encontrada
      for edge in edges_cursor:
        involved_node_ids.add(edge['source'])
        involved_node_ids.add(edge['target'])
        
        edge_info = {
          "source": edge["source"],
          "target": edge['target'],
          "relation_type": edge["relation_type"],
          "properties": edge["properties"],
          "directed": edge["directed"]
        }
        subgraph_edges.append(edge_info)
      
      # si no se encontraron aristas del tipo específico 
      if not subgraph_edges:
        return {
          "success": True,
          "message": f"No se encontraron relaciones del tipo: {relation_type}",
          "nodes": [],
          "edges": [],
          "stats": {
            "total_nodes": 0,
            "total_edges": 0,
            "relation_type": relation_type
          }
        }
      
      # obtener información de todos los nodos involucrados
      subgraph_nodes = []
      nodes_cursor = self.nodes.find({"node_id": {"$in": list(involved_node_ids)}})
      
      for node in nodes_cursor: 
        node_info = {
          "node_id": node["node_id"],
          "type": node["type"],
          "properties": node["properties"]
        }
        subgraph_nodes.append(node_info)
      
      # generar estadísticas del subgrafo
      node_types = list(set(node["type"] for node in subgraph_nodes))
      
      stats = {
        "total_nodes": len(subgraph_nodes),
        "total_edges": len(subgraph_edges),
        "relation_type": relation_type,
        "node_types_in_subgraph": node_types,
        "unique_nodes": len(involved_node_ids),
        "directed_edges": sum(1 for edge in subgraph_edges if edge["directed"]),
        "undirected_edges": sum(1 for edge in subgraph_edges if not edge["directed"])
      }
      
      return {
        "success": True,
        "message": f"Subgrafo extraído exitosamente para relación: {relation_type}",
        "nodes": subgraph_nodes,
        "edges": subgraph_edges,
        "stats": stats 
      }
    except Exception as e:
      return {
        "success": False,
        "message": f"Error al extraer subgrafo: {str(e)}",
        "nodes": [],
        "edges": [],
        "stats": {
          "total_nodes": 0,
          "total_edges": 0,
          "relation_type": relation_type
        }
      }

  def is_dag(self) -> Dict:
    """Verifica si el grafo es un DAG (Directed Acyclic Graph)

    Un grafo es DAG si: 
    1. Todas las aristas son dirigidas
    2. No contiene ciclos

    Utiliza el algoritmo de Kahn (orden topológico) para detectar ciclos

    Returns:
      Dict: resultado de la verificación:
        - `is_dag`: bool indicando si es DAG
        - `message`: str con mensaje descriptivo
        - `has_undirected_edges`: bool si tiene aristas no dirigidas
        - `has_cycles`: bool si tiene ciclos
        - `cycle_info`: Dict con información del ciclo encontrado (si existe)
        - `stats`: Dict con estadísticas del análisis
    """
    try:
      # obtener todas las aristas
      edges_cursor = self.edges.find({})
      edges_list = list(edges_cursor)
      
      # verificar si hay aristas no dirigdas
      undirected_edges = [edge for edge in edges_list if not edge.get("directed", True)]
      has_undirected = len(undirected_edges) > 0
      
      if has_undirected:
        return {
          "is_dag": False,
          "message": f"El grafo NO es DAG: contiene {len(undirected_edges)} aristas no dirigidas",
          "has_undirected_edges": True,
          "has_cycles": None,   # no se puede determinar sin verificar ciclos
          "cycle_info": None,
          "stats": {
            "total_edges": len(edges_list),
            "directed_edges": len(edges_list) - len(undirected_edges),
            "undirected_edges": len(undirected_edges),
            "total_nodes": self.nodes.count_documents({})
          }
        }
        
      # solo considerar aristas dirigidas para el análisis de ciclos
      directed_edges = [edge for edge in edges_list if edge.get("directed", True)]
      cycle_result = self._detect_cycles_kahn(directed_edges)
      
      if cycle_result["has_cycles"]:
        return {
          "is_dag": False, 
          "message": f"El grafo NO es DAG: se detectaron ciclos",
          "has_undirected_edges": False,
          "has_cycles": True,
          "cycle_info": cycle_result["cycle_info"],
          "stats": {
            "total_edges": len(edges_list),
            "directed_edges": len(directed_edges),
            "undirected_edges": len(undirected_edges),
            "total_nodes": self.nodes.count_documents({}),
            "nodes_in_cycles": len(cycle_result["cycle_info"]["remaining_nodes"])
          }
        }
      return {
        "is_dag": True,
        "message": "El grafo es DAG: todas las aristas son dirigidas y no hay ciclos",
        "has_undirected_edges": False,
        "has_cycles": False,
        "cycle_info": None,
        "stats": {
          "total_edges": len(edges_list),
          "directed_edges": len(directed_edges),
          "undirected_edges": len(undirected_edges),
          "total_nodes": self.nodes.count_documents({}),
          "topological_order": cycle_result["topological_order"]
        }
      }
    except Exception as e:
      return {
        "is_dag": False,
        "message": f"Error al verificar DAG: {str(e)}",
        "has_undirected_edges": None,
        "has_cycles": None,
        "stats": {}
      }

  def _detect_cycles_kahn(self, directed_edges:List) -> Dict:
    """Detecta ciclos usando el algoritmo de Kahn (orden topológico)
    
    Args:
      directed_edges (List): lista de aristas dirigidas

    Returns:
      Dict: resultado de análisis de ciclos
    """
    try: 
      # Construir grafo de adyacencia y contar grados de entrada
      graph = {}
      in_degree = {}
      all_nodes = set()
      
      # Inicializar estructuras
      for edge in directed_edges:
        source = edge["source"]
        target = edge["target"]
        all_nodes.add(source)
        all_nodes.add(target)
        
        if source not in graph:
          graph[source] = []
        if target not in graph:
          graph[target] = []
        if source not in in_degree:
          in_degree[source] = 0
        if target not in in_degree:
          in_degree[target] = 0
      
      # Construir lista de adyacencia y contar grados de entrada
      for edge in directed_edges:
        source = edge["source"]
        target = edge["target"]
        graph[source].append(target)
        in_degree[target] += 1
      
      # Encontrar nodos con grado de entrada 0
      queue = [node for node in all_nodes if in_degree[node] == 0]
      topological_order = []
      
      while queue:
        current = queue.pop(0)
        topological_order.append(current)
        
        # Reducir grado de entrada de nodos adyacentes
        for neighbor in graph[current]:
          in_degree[neighbor] -= 1
          if in_degree[neighbor] == 0:
            queue.append(neighbor)
      
      # Si no todos los nodos están en el orden topológico, hay ciclos
      if len(topological_order) != len(all_nodes):
        remaining_nodes = [node for node in all_nodes if node not in topological_order]
        
        # Encontrar un ciclo específico
        cycle_path = self._find_cycle_path(graph, remaining_nodes)
        
        return {
          "has_cycles": True,
          "cycle_info": {
            "remaining_nodes": remaining_nodes,
            "nodes_in_cycles": len(remaining_nodes),
            "example_cycle": cycle_path
          },
          "topological_order": topological_order
        }
      else:
        return {
          "has_cycles": False,
          "cycle_info": None,
          "topological_order": topological_order
        }
    except Exception as e:
      return {
        "has_cycles": None,
        "cycle_info": {"error": str(e)},
        "topological_order": []
      }

  def find_paths_between_entities(self,
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

  def calculate_relation_probabilities(self,
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

  def calculate_indirect_probability(self,
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

  def build_path_details(self, path:List[str]) -> List[Dict]:
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

  def retriever(self, 
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

#region TOOLS
def add_prerequisites_from_map(graph:GraphDB, prerequisite_map:Dict[str,List[str]]) -> None:
  """Inserta aristas de tipo 'prerequisito' basadas en eun diccionario donde:
  - key: subtema prerrequisito
  - value: lista de subtemas que lo requieren

  Args:
    graph (GraphDB): instancia de GraphDB
    prerequisite_map (Dict): diccionario con subtemas y listas de dependencias  
  """
  total = 0
  for prerequisite, topics in prerequisite_map.items():
    for topic in topics:
      result = graph.add_edge(
        source_id=prerequisite,
        target_id=topic,
        relation_type="prerequisito",
        properties={"explicación": f"{prerequisite} es prerequisito de {topic}"},
        directed=True
      )
      total += 1 if result["success"] else 0 
      status = "OK" if result["success"] else "ERROR"
      print(f"{status}: {prerequisite} -> {topic} : {result['message']}")

def visualize_graph_matplotlib(graph_db, layout_type="spring", 
      node_size_factor=1000, 
      edge_width_factor=2,
      figsize=(12, 8),
      show_labels=True,
      filter_node_type=None,
      filter_relation_type=None):
  """
  Visualiza el grafo usando matplotlib/networkx
  
  Args:
    graph_db: Instancia de GraphDB
    layout_type: Tipo de layout ('spring', 'circular', 'random', 'shell')
    node_size_factor: Factor de tamaño de los nodos
    edge_width_factor: Factor de grosor de las aristas
    figsize: Tamaño de la figura
    show_labels: Si mostrar etiquetas en los nodos
    filter_node_type: Filtrar por tipo de nodo específico
    filter_relation_type: Filtrar por tipo de relación específica
  """
  
  # Crear grafo NetworkX
  G = nx.DiGraph()
  
  # Obtener todos los nodos
  nodes = graph_db.get_all_nodes()
  
  # Filtrar nodos si se especifica
  if filter_node_type:
        nodes = [node for node in nodes if node['type'] == filter_node_type]
    
  # Añadir nodos al grafo
  node_colors = {}
  node_types = list(set([node['type'] for node in nodes]))
  colors = plt.cm.Set3(np.linspace(0, 1, len(node_types)))
    
  for i, node_type in enumerate(node_types):
    node_colors[node_type] = colors[i]
    
  for node in nodes:
    G.add_node(node['node_id'], type=node['type'], properties=node['properties'])
  
  # Obtener todas las aristas
  edges_cursor = graph_db.edges.find({})
  
  for edge in edges_cursor:
    # Filtrar por tipo de relación si se especifica
    if filter_relation_type and edge['relation_type'] != filter_relation_type:
      continue
    
    # Solo añadir arista si ambos nodos están en el grafo filtrado
    if edge['source'] in G.nodes() and edge['target'] in G.nodes():
      G.add_edge(edge['source'], edge['target'], relation_type=edge['relation_type'], properties=edge['properties'], directed=edge['directed'])
  
  # Configurar el layout
  match layout_type:
    case "spring":
      pos = nx.spring_layout(G, k=1, iterations=50)
    case "circular":
      pos = nx.circular_layout(G)
    case "random":
      pos = nx.random_layout(G)
    case "shell":
      pos = nx.shell_layout(G)
    case _:
      pos = nx.spring_layout(G)
  
  # Crear la figura
  fig, ax = plt.subplots(figsize=figsize)
    
  # Dibujar nodos por tipo (para diferentes colores)
  for node_type in node_types:
    node_list = [node for node in G.nodes() if G.nodes[node]['type'] == node_type]
    nx.draw_networkx_nodes(G, pos, 
      nodelist=node_list, 
      node_color=[node_colors[node_type]], 
      node_size=node_size_factor,
      alpha=0.8, 
      ax=ax
    )
    
  # Dibujar aristas
  nx.draw_networkx_edges(G, pos, 
    edge_color='gray', 
    width=edge_width_factor, 
    alpha=0.6, 
    arrows=True, 
    arrowsize=20, 
    ax=ax)
  
  # Dibujar etiquetas si se solicita
  if show_labels:
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
  
  # Crear leyenda
  legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=node_colors[node_type], 
                      markersize=10, label=node_type)
                    for node_type in node_types]
  ax.legend(handles=legend_elements, loc='upper right')
  
  ax.set_title(f"Grafo de Conocimiento - {len(G.nodes())} nodos, {len(G.edges())} aristas")
  ax.axis('off')
  
  return fig

def visualize_graph_plotly(graph_db, 
      layout_type="spring",
      node_size_factor=20,
      filter_node_type=None,
      filter_relation_type=None,
      show_edge_labels=False):
  """
  Visualiza el grafo usando Plotly (interactivo)
    
  Args:
    graph_db: Instancia de GraphDB
    layout_type: Tipo de layout
    node_size_factor: Factor de tamaño de los nodos
    filter_node_type: Filtrar por tipo de nodo específico
    filter_relation_type: Filtrar por tipo de relación específica
    show_edge_labels: Si mostrar etiquetas en las aristas
  """
  
  # Crear grafo NetworkX
  G = nx.DiGraph()
  
  # Obtener todos los nodos
  nodes = graph_db.get_all_nodes()
  
  # Filtrar nodos si se especifica
  if filter_node_type:
    nodes = [node for node in nodes if node['type'] == filter_node_type]
  
  # Añadir nodos al grafo
  for node in nodes:
    G.add_node(node['node_id'], type=node['type'], properties=node['properties'])
  
  # Obtener todas las aristas
  edges_cursor = graph_db.edges.find({})
    
  for edge in edges_cursor:
    # Filtrar por tipo de relación si se especifica
    if filter_relation_type and edge['relation_type'] != filter_relation_type:
      continue
    
    # Solo añadir arista si ambos nodos están en el grafo filtrado
    if edge['source'] in G.nodes() and edge['target'] in G.nodes():
      G.add_edge(edge['source'], edge['target'], relation_type=edge['relation_type'], properties=edge['properties'], directed=edge['directed'])
  
  match layout_type:
    case "spring":
      pos = nx.spring_layout(G, k=1, iterations=50)
    case "circular":
      pos = nx.circular_layout(G) 
    case _:
      pos = nx.spring_layout(G)   
  
  # Preparar datos para Plotly
  node_trace = go.Scatter(
    x=[], 
    y=[], 
    mode='markers+text', 
    hoverinfo='text', 
    text=[], textposition="middle center",
    marker=dict(size=[], color=[], 
      colorscale='Viridis', 
      showscale=True,
      colorbar=dict(title="Tipo de Nodo")
    )
  )
  
  # Crear mapeo de tipos de nodo a números para colorear
  node_types = list(set([G.nodes[node]['type'] for node in G.nodes()]))
  type_to_num = {node_type: i for i, node_type in enumerate(node_types)}
  
  # Añadir nodos
  for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple([node])
    node_trace['marker']['size'] += tuple([node_size_factor])
    node_trace['marker']['color'] += tuple([type_to_num[G.nodes[node]['type']]])
  
  # Crear hover text para nodos
  node_info = []
  for node in G.nodes():
    info = f"ID: {node}<br>Tipo: {G.nodes[node]['type']}<br>"
    if G.nodes[node]['properties']:
      info += f"Propiedades: {G.nodes[node]['properties']}"
    node_info.append(info)
  
  node_trace['hovertext'] = node_info
  
  # Preparar aristas
  edge_trace = go.Scatter(x=[], y=[], mode='lines', line=dict(width=2, color='gray'), hoverinfo='none')
  
  edge_info = []
  for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    
    # Información de la arista
    if show_edge_labels:
      edge_info.append(f"{edge[0]} -> {edge[1]}: {G.edges[edge]['relation_type']}")
  
  # Crear la figura
  fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(title='Grafo de Conocimiento',
      showlegend=False,
      hovermode='closest',
      margin=dict(b=20,l=5,r=5,t=40),
      annotations=[ dict(
      text="Interactivo: arrastra para mover, zoom con rueda del mouse",
      showarrow=False,
      xref="paper", yref="paper",
      x=0.005, y=-0.002,
      xanchor='left', yanchor='bottom',
      font=dict(size=12)
    )],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
  )
  
  return fig

def display_graph_statistics(graph_db):
  "Muestra estadísticas del grafo en Streamlit"
  stats = graph_db.get_graph_stats()
  
  col1, col2, col3, col4 = st.columns(4)
  
  with col1:
    st.metric("Total Nodos", stats['total_nodes'])
  with col2:
    st.metric("Total Aristas", stats['total_edges'])
  with col3:
    st.metric("Tipos de Nodos", len(stats['node_types']))
  with col4:
    st.metric("Tipos de Relaciones", len(stats['relationship_types']))
  
  # Mostrar detalles
  col1, col2 = st.columns(2)
  
  with col1:
    with st.expander("Tipos de Nodos"):
      for node_type in stats['node_types']:
        count = graph_db.nodes.count_documents({"type": node_type})
        st.write(f"• {node_type}: {count}")
  
  with col2:
    with st.expander("Tipos de Relaciones"):
      for rel_type in stats['relationship_types']:
        count = graph_db.edges.count_documents({"relation_type": rel_type})
        st.write(f"• {rel_type}: {count}")

def streamlit_graph_interface(graph_db):
  "Interfaz completa de Streamlit para visualizar el grafo"
  
  st.title("🕸️ Visualización del Grafo de Conocimiento")
  
  # Mostrar estadísticas
  st.header("📊 Estadísticas del Grafo")
  display_graph_statistics(graph_db)
  
  # Configuración de visualización
  st.header("🎨 Configuración de Visualización")
  
  col1, col2 = st.columns(2)
    
  with col1:
    viz_type = st.selectbox("Tipo de Visualización", 
      ["Plotly (Interactivo)", "Matplotlib (Estático)"])
        
    layout_type = st.selectbox("Tipo de Layout", 
      ["spring", "circular", "random", "shell"])
    
  with col2:
    # Filtros
    node_types = graph_db.get_node_types()
    filter_node_type = st.selectbox("Filtrar por Tipo de Nodo", 
      ["Todos"] + node_types)
        
    relation_types = graph_db.get_relationship_types()
    filter_relation_type = st.selectbox("Filtrar por Tipo de Relación", 
      ["Todas"] + relation_types)
    
  # Aplicar filtros
  node_filter = None if filter_node_type == "Todos" else filter_node_type
  relation_filter = None if filter_relation_type == "Todas" else filter_relation_type
    
  # Generar visualización
  if st.button("🔄 Generar Visualización"):
    with st.spinner("Generando visualización..."):
    
      if viz_type == "Plotly (Interactivo)":
        fig = visualize_graph_plotly(graph_db, 
          layout_type=layout_type,
          filter_node_type=node_filter,
          filter_relation_type=relation_filter)
        st.plotly_chart(fig, use_container_width=True)

      else:  # Matplotlib
        fig = visualize_graph_matplotlib(graph_db,
          layout_type=layout_type,
          filter_node_type=node_filter,
          filter_relation_type=relation_filter)
        st.pyplot(fig)
  
  # Sección de exploración de nodos
  st.header("🔍 Exploración de Nodos")
  
  # Seleccionar nodo para explorar
  all_nodes = graph_db.get_all_nodes()
  node_ids = [node['node_id'] for node in all_nodes]
  
  if node_ids:
    selected_node = st.selectbox("Seleccionar Nodo para Explorar", node_ids)
    
    if st.button("🕵️ Explorar Nodo"):
      # Mostrar información del nodo
      node_info = graph_db.get_node_info(selected_node)
      st.subheader(f"Información del Nodo: {selected_node}")
      st.json(node_info)
      
      # Mostrar entidades adyacentes
      adjacent = graph_db.get_adjacent_entities(selected_node)
      if adjacent:
        st.subheader("Entidades Adyacentes")

        # Crear dataframe para mostrar mejor
        adj_data = []
        for adj in adjacent:
          adj_data.append({
            'Nodo Adyacente': adj['node_id'],
            'Tipo': adj['type'],
            'Relación': adj['relationship']['type'],
            'Dirección': adj['relationship']['direction']
          })
        
        df = pd.DataFrame(adj_data)
        st.dataframe(df)
      else:
        st.info("Este nodo no tiene entidades adyacentes")


