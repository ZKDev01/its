""" Creacion de vectorstore y graphDB a partir de únicamente documentos
1. Procesar los documentos (pdf) a markdown
1.1. Obtener por cada archivo markdown los chunks 

# Vectorstore
1.1.1. Obtener de cada ckunks la tripleta <C,S,QA>
      - C: chunk original
      - S: resumen de lo que dice el chunk
      - QA: preguntas y respuesta utilizando el mismo chunk

# Graph DB
1.1.2. Extraer de cada chunk las entidades y relaciones
    - Entidades: primero se analizan las entidades
    - Relaciones: luego se pregunta al LLM cuales entidades estan relacionadas y como

1.2. Salvar cada resultado en DB diferentes
1.3. <Routing> 
"""