import streamlit as st
from src.knowledge_graph import (
  GraphDB,
  streamlit_graph_interface,
) 

# Configuración de la página
st.set_page_config(
  page_title="Graph Database Visualizer",
  page_icon="🕸️",
  layout="wide"
)

def main():
  # Inicializar conexión a la base de datos
  @st.cache_resource
  def init_database():
    return GraphDB()
    
  try:
    graph_db = init_database()
    
    # Sidebar para navegación
    st.sidebar.title("🕸️ Graph DB Visualizer")
    
    menu_option = st.sidebar.selectbox("Selecciona una opción:", ["Visualizar Grafo", "Gestionar Nodos", "Gestionar Relaciones"])
    
    match menu_option:
      case "Visualizar Grafo":
        streamlit_graph_interface(graph_db)
      case "Gestionar Nodos":
        manage_nodes_interface(graph_db)
      case "Gestionar Relaciones":
        manage_edges_interface(graph_db)
  
  except Exception as e:
    st.info(f"ERROR: {str(e)}")

def manage_nodes_interface(graph_db):
  "Interfaz para gestionar nodos"
  st.title("🎯 Gestión de Nodos")
  
  tab1, tab2, tab3 = st.tabs(["Añadir Nodo", "Ver Nodos", "Eliminar Nodo"])
  
  with tab1:
    st.subheader("Añadir Nuevo Nodo")
    
    with st.form("add_node_form"):
      node_id = st.text_input("ID del Nodo", placeholder="ej: persona_001")
      node_type = st.text_input("Tipo de Nodo", placeholder="ej: Persona")
      
      # Propiedades adicionales
      st.write("Propiedades Adicionales:")
      prop_key = st.text_input("Clave de Propiedad", placeholder="ej: edad")
      prop_value = st.text_input("Valor de Propiedad", placeholder="ej: 25")
      
      if st.form_submit_button("Añadir Nodo"):
        properties = {prop_key: prop_value} if prop_key and prop_value else {}

        if node_id and node_type:
          success = graph_db.add_node(node_id, node_type, properties)
          if success:
            st.success(f"Nodo '{node_id}' añadido exitosamente!")
          else:
            st.error("Error al añadir el nodo")
        else:
          st.error("Por favor completa los campos obligatorios")
    
  with tab2:
    st.subheader("Nodos Existentes")
        
    nodes = graph_db.get_all_nodes()
    if nodes:
      # Mostrar como tabla
      node_data = []
      for node in nodes:
        node_data.append( 
          {
            'ID': node['node_id'],
            'Tipo': node['type'],
            'Propiedades': str(node['properties']) if node['properties'] else 'Ninguna'
          }
        )
      
      import pandas as pd
      df = pd.DataFrame(node_data)
      st.dataframe(df, use_container_width=True)
    else:
      st.info("No hay nodos en la base de datos")
  
  with tab3:
    st.subheader("Eliminar Nodo")
    
    nodes = graph_db.get_all_nodes()
    if nodes:
      node_ids = [node['node_id'] for node in nodes]
      selected_node = st.selectbox("Seleccionar Nodo a Eliminar", node_ids)
      
      cascade = st.checkbox("Eliminar también las relaciones (cascade)", value=True)
      
      if st.button("🗑️ Eliminar Nodo", type="primary"):
        result = graph_db.del_node(selected_node, cascade=cascade)
        
        if result['success']:
          st.success(f"Nodo eliminado: {result['message']}")
          if result['edges_deleted'] > 0:
            st.info(f"También se eliminaron {result['edges_deleted']} relaciones")
        else:
          st.error(f"Error: {result['message']}")
    else:
      st.info("No hay nodos para eliminar")

def manage_edges_interface(graph_db):
  "Interfaz para gestionar relaciones"
  st.title("🔗 Gestión de Relaciones")
  
  tab1, tab2 = st.tabs(["Añadir Relación", "Ver Relaciones"])
  
  with tab1:
    st.subheader("Añadir Nueva Relación")
    
    nodes = graph_db.get_all_nodes()
    if len(nodes) >= 2:
      node_ids = [node['node_id'] for node in nodes]
      
      with st.form("add_edge_form"):
        source = st.selectbox("Nodo Origen", node_ids)
        target = st.selectbox("Nodo Destino", node_ids)
        relation_type = st.text_input("Tipo de Relación", placeholder="ej: conoce")
        
        directed = st.checkbox("Relación Dirigida", value=True)
        
        # Propiedades adicionales
        st.write("Propiedades Adicionales:")
        prop_key = st.text_input("Clave", placeholder="ej: desde")
        prop_value = st.text_input("Valor", placeholder="ej: 2020")
        
        if st.form_submit_button("Añadir Relación"):
          properties = {prop_key: prop_value} if prop_key and prop_value else {}
          
          if source and target and relation_type:
            if source != target:
              result = graph_db.add_edge(source, target, relation_type, properties, directed)
              
              if result['success']:
                st.success(f"Relación añadida: {result['message']}")
              else:
                st.error(f"Error: {result['message']}")
            else:
              st.error("El nodo origen y destino no pueden ser el mismo")
          else:
            st.error("Por favor completa todos los campos")
    else:
      st.info("Necesitas al menos 2 nodos para crear una relación")
  
  with tab2:
    st.subheader("Relaciones Existentes")
    
    edges_cursor = graph_db.edges.find({})
    edges = list(edges_cursor)
    
    if edges:
      edge_data = []
      for edge in edges:
        edge_data.append({
          'Origen': edge['source'],
          'Destino': edge['target'],
          'Tipo': edge['relation_type'],
          'Dirigida': 'Sí' if edge['directed'] else 'No',
          'Propiedades': str(edge['properties']) if edge['properties'] else 'Ninguna'
        })
        
      import pandas as pd
      df = pd.DataFrame(edge_data)
      st.dataframe(df, use_container_width=True)
    else:
      st.info("No hay relaciones en la base de datos")


if __name__ == "__main__":
  main()
