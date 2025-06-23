import streamlit as st

from src.graph_retriever import *

GraphDB.retriever = retriever
GraphDB._find_paths_between_entities = find_paths_between_entities
GraphDB._calculate_relation_probabilities = calculate_relation_probabilities
GraphDB._calculate_indirect_probability = calculate_indirect_probability
GraphDB._build_path_details = build_path_details

graphDB = GraphDB()

# Realiza una consulta usando el retriever de graphDB
nodes = [ node["node_id"] for node in graphDB.get_all_nodes() ]
query_entity_1 = st.selectbox("Selecciona la entidad de origen:", nodes)
query_entity_2 = st.selectbox("Selecciona la entidad de destino (opcional):", nodes + [None])

if st.button("Realizar consulta"):
  if query_entity_2 is None:
    results = graphDB.retriever(query_entity_1)
  else:
    results = graphDB.retriever(query_entity_1, query_entity_2)
  st.write("Resultados de la consulta:")
  st.write(results)