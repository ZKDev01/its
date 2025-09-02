import streamlit as st  

from src.controller import (
    AgentEngine, Ollama, 
    TaskPlannerAgent, TextGenAgent, ReasoningAgent, QAGenerationAgent
)
from src.ga_ils_scheduling_systems import Solution, load_data_from_json

# Inicializar LLM y AgentEngine
llm = Ollama(model="gemma3:1b")
engine = AgentEngine(llm)

# Registrar los 4 agentes si no están registrados
if not engine.task_manager.agents:
    docs = ["Este es un documento de ejemplo para pruebas."]
    engine.register_agent(TaskPlannerAgent(llm=llm))
    engine.register_agent(TextGenAgent(llm=llm, documents=docs))
    engine.register_agent(ReasoningAgent(llm=llm))
    #engine.register_agent(QAGenerationAgent(llm=llm))

st.title("Intelligent Tutoring System - AgentEngine")

query = st.text_input("Introduce tu problema o pregunta aquí")
if st.button("Procesar"):
    response = engine.run_cycle(query)
    st.markdown(f"**Respuesta:**\n\n{response.result}")
    st.info(f"Agente utilizado: {response.agent}")
    st.success(f"Estado: {'Éxito' if response.success else 'Error'}")
    if response.message:
        st.caption(f"Mensaje: {response.message}")

    # Mostrar la Solution como tabla si existe
    solution = response.get_data("solution", None)
    if solution is not None and hasattr(solution, "assignments"):
        try:
            # Cargar los datos para obtener las descripciones de las tareas
            tasks, days, turns_per_day = load_data_from_json("generated_data.json")
            task_desc_map = {task.id: task.description for task in tasks}

            # Obtener dimensiones (ajustar para que empiece en 1)
            max_day = max(a.day for a in solution.assignments)
            max_turn = max(a.start_turn for a in solution.assignments)

            # La tabla será de tamaño [maxTurn][maxDay], filas=turnos, columnas=días
            matrix = [[None for _ in range(1, max_day + 1)] for _ in range(1, max_turn + 1)]
            for a in solution.assignments:
                # Usar índices ajustados para que la tabla empiece en 1
                matrix[a.start_turn - 1][a.day - 1] = a.task_id

            st.markdown("### Tabla de asignación de tareas (Solution)")
            st.write("Cada celda representa el ID de la tarea asignada para ese turno y día (columnas=días, filas=turnos, comenzando en 1).")
            st.table(matrix)

            # Leyenda de tareas
            st.markdown("#### Leyenda de tareas")
            for tid, desc in task_desc_map.items():
                st.markdown(f"- **{tid}**: {desc}")

        except Exception as e:
            st.warning(f"No se pudo mostrar la solución en formato de tabla: {e}")



