import os
import random
import json

def generar_examen(numero_examen, nivel=0):
    base_dir = os.path.join("..","resources","datasets", "MATH", "train")
    examen = {}
    # Recorre cada subcarpeta
    for carpeta in os.listdir(base_dir):
        carpeta_path = os.path.join(base_dir, carpeta)
        if os.path.isdir(carpeta_path):
            archivos = [f for f in os.listdir(carpeta_path) if f.endswith('.json')]
            preguntas_filtradas = []
            for archivo in archivos:
                archivo_path = os.path.join(carpeta_path, archivo)
                with open(archivo_path, 'r', encoding='utf-8') as f:
                    pregunta = json.load(f)
                    if nivel == 0 or pregunta.get("level") == f"Level {nivel}":
                        preguntas_filtradas.append(pregunta)
            if preguntas_filtradas:
                pregunta_elegida = random.choice(preguntas_filtradas)
                examen[carpeta] = pregunta_elegida
    
    # Guarda el examen en la carpeta objetivo, si no existe, la crea
    folder_path = os.path.join("..", "exams")
    out_path = os.path.join(folder_path, f"Examen {numero_examen}.json")

    os.makedirs(folder_path, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(examen, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    generar_examen(1, nivel=1)  # Elige nivel de dificultad (0 - 5), 0 para todos
