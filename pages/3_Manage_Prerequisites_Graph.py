import streamlit as st
from src.knowledge_graph import (
  GraphDB, 
  streamlit_graph_interface
)

st.set_page_config(
  page_title="Prerequisites Graph Visualizer",
  page_icon="🕸️", 
  layout="wide"
)

def main():
  @st.cache_resource
  def init_database():
    return GraphDB("prerequisites_graph")
  
  try:
    graph_db = init_database()
    streamlit_graph_interface(graph_db)
  except Exception as e:
    st.info(f"ERROR: {str(e)}")

# prerequisites_graph = GraphDB("prerequisites_graph")

if __name__ == "__main__":
  main()

