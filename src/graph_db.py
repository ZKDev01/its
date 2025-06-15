import json 
from pymongo import MongoClient
from typing import List, Dict, Optional




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
  
  def retriever(self) -> ... :
    pass 
  
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