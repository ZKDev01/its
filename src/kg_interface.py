from src.knowledge_graph import GraphDB, add_prerequisites_from_map
  

# instanciar y resetar para prueba
graph = GraphDB("prerequisites_graph")
graph.reset_database(True)

# definición de los tipos de tópicos y su clasificación
type_map = {
  "algebra": ["linear_equations", "quadratic_equations", "polynomials", "matrices", "vector_spaces"],
  "analysis": ["limits", "derivatives", "integrals", "sequences", "series"]
}

for key, topics in type_map.items():
  for topic in topics:
    graph.add_node(topic, key) 

# definición de los prerequisitos
prerequisite_map = {
  "limits": ["derivatives", "series", "polynomials"],
  "derivatives": ["integrals"],
  "sequences": ["series"],
  "polynomials": ["quadratic_equations", "vector_spaces"],
  "matrices": ["linear_equations", "vector_spaces"],
  "vector_spaces": ["matrices"],
  "linear_equations": ["matrices"],
  "quadratic_equations": ["polynomials"],
  "series": ["limits"],
  "integrals": ["derivatives"]
}

add_prerequisites_from_map(graph=graph, prerequisite_map=prerequisite_map)
print()
result = graph.is_dag()
for key, value in result.items():
  print(f"[{key}] => {value}")
