import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy

@dataclass
class Parametros:
    """Parámetros del sistema de planificación"""
    # Parámetros del algoritmo genético
    tamaño_poblacion: int = 100
    generaciones: int = 500
    prob_cruce: float = 0.85
    prob_mutacion: float = 0.15
    elite_porcentaje: float = 0.10
    
    # Parámetros de evaluación
    w_S: float = 0.3  # Peso espaciamiento
    w_P: float = 0.5  # Peso preparación exámenes
    w_C: float = 0.2  # Peso balance de carga
    umbral_carga: float = 0.8  # Umbral de sobrecarga diaria
    epsilon: float = 0.01  # Umbral para considerar planes equivalentes

class PlanificadorEstudios:
    """Sistema principal de planificación de estudios"""
    
    def __init__(self, n_dias: int, n_turnos: int, asignaturas: List[str], 
                 parametros: Parametros = None):
        self.n_dias = n_dias
        self.n_turnos = n_turnos
        self.M = len(asignaturas)  # Número de asignaturas
        self.asignaturas = asignaturas
        self.params = parametros or Parametros()
        
        # Inicializar matrices y vectores
        self.e_minimos = np.zeros(self.M)  # Sesiones mínimas por asignatura
        self.disponibilidad = np.ones((n_dias, n_turnos))  # Disponibilidad estudiante
        self.preferencia = np.ones((self.M, n_turnos)) * 3  # Preferencias (1-5)
        self.dificultad = np.ones(self.M) * 5  # Dificultades (1-10)
        self.prioridad = np.ones((self.M, n_dias)) * 0.5  # Prioridades
        self.calendario_examenes = np.zeros((n_dias + 1, self.M))  # Matriz de exámenes
        
    def configurar_disponibilidad(self, disponibilidad: np.ndarray):
        """Configurar disponibilidad del estudiante"""
        self.disponibilidad = disponibilidad
        
    def configurar_preferencias(self, preferencias: np.ndarray):
        """Configurar preferencias por turno y asignatura"""
        self.preferencia = preferencias
        
    def configurar_dificultades(self, dificultades: np.ndarray):
        """Configurar dificultades de asignaturas"""
        self.dificultad = dificultades
        
    def configurar_sesiones_minimas(self, sesiones_minimas: np.ndarray):
        """Configurar sesiones mínimas por asignatura"""
        self.e_minimos = sesiones_minimas
        
    def configurar_examenes(self, calendario_examenes: np.ndarray):
        """Configurar calendario de exámenes"""
        self.calendario_examenes = calendario_examenes

class Cromosoma:
    """Representación de una solución (plan de estudios)"""
    
    def __init__(self, n_dias: int, n_turnos: int, M: int):
        self.n_dias = n_dias
        self.n_turnos = n_turnos
        self.M = M
        self.matriz = np.zeros((n_dias, n_turnos), dtype=int)
        self.fitness = 0.0
        
    def generar_aleatorio(self, planificador: PlanificadorEstudios, alpha: float = 0.7):
        """Generar cromosoma inicial usando método constructivo"""
        for j in range(self.n_dias):
            for k in range(self.n_turnos):
                if planificador.disponibilidad[j, k] == 1:
                    # Calcular probabilidades ponderadas
                    probs = []
                    for i in range(self.M):
                        p_i = planificador.prioridad[i, j]
                        pref_i_k = planificador.preferencia[i, k]
                        prob = alpha * p_i * pref_i_k
                        probs.append(prob)
                    
                    # Normalizar probabilidades
                    if sum(probs) > 0:
                        probs = np.array(probs) / sum(probs)
                        asignatura = np.random.choice(self.M, p=probs) + 1
                        self.matriz[j, k] = asignatura
                    else:
                        self.matriz[j, k] = 0
        
        # Reparar solución para cumplir sesiones mínimas
        self._reparar_sesiones_minimas(planificador)
    
    def _reparar_sesiones_minimas(self, planificador: PlanificadorEstudios):
        """Reparar cromosoma para cumplir sesiones mínimas"""
        for i in range(self.M):
            sesiones_actuales = np.sum(self.matriz == i + 1)
            sesiones_requeridas = planificador.e_minimos[i]
            
            if sesiones_actuales < sesiones_requeridas:
                # Encontrar turnos disponibles donde reemplazar
                deficit = int(sesiones_requeridas - sesiones_actuales)
                turnos_disponibles = []
                
                for j in range(self.n_dias):
                    for k in range(self.n_turnos):
                        if (planificador.disponibilidad[j, k] == 1 and 
                            self.matriz[j, k] != 0):
                            # Priorizar reemplazar asignaturas con baja prioridad
                            asig_actual = self.matriz[j, k] - 1
                            if asig_actual >= 0:
                                prioridad_actual = planificador.prioridad[asig_actual, j]
                                turnos_disponibles.append((j, k, prioridad_actual))
                
                # Ordenar por prioridad ascendente y reemplazar
                turnos_disponibles.sort(key=lambda x: x[2])
                for idx in range(min(deficit, len(turnos_disponibles))):
                    j, k, _ = turnos_disponibles[idx]
                    self.matriz[j, k] = i + 1
    
    def copiar(self):
        """Crear copia del cromosoma"""
        nuevo = Cromosoma(self.n_dias, self.n_turnos, self.M)
        nuevo.matriz = self.matriz.copy()
        nuevo.fitness = self.fitness
        return nuevo

class AlgoritmoGenetico:
    """Implementación del algoritmo genético"""
    
    def __init__(self, planificador: PlanificadorEstudios):
        self.planificador = planificador
        self.poblacion = []
        self.mejor_solucion = None
        self.historial_fitness = []
        
    def inicializar_poblacion(self):
        """Inicializar población usando método constructivo"""
        self.poblacion = []
        for _ in range(self.planificador.params.tamaño_poblacion):
            cromosoma = Cromosoma(self.planificador.n_dias, 
                                self.planificador.n_turnos, 
                                self.planificador.M)
            cromosoma.generar_aleatorio(self.planificador)
            self.poblacion.append(cromosoma)
    
    def evaluar_fitness(self, cromosoma: Cromosoma) -> float:
        """Calcular fitness usando función objetivo original"""
        evaluador = EvaluadorPlanes(self.planificador)
        S, P, C = evaluador._calcular_metricas_cromosoma(cromosoma)
        
        # Función objetivo original
        f_objetivo = -(self.planificador.params.w_S * S + 
                      self.planificador.params.w_P * P + 
                      self.planificador.params.w_C * C)
        
        # Fitness transformado
        fitness = 1.0 / (1.0 + abs(f_objetivo))
        cromosoma.fitness = fitness
        return fitness
    
    def seleccion_torneo(self, k: int = 2) -> Cromosoma:
        """Selección por torneo binario"""
        competidores = random.sample(self.poblacion, k)
        return max(competidores, key=lambda x: x.fitness)
    
    def cruce_bloques_dias(self, padre1: Cromosoma, padre2: Cromosoma) -> Cromosoma:
        """Cruce por intercambio de bloques de días"""
        hijo = padre1.copiar()
        
        # Punto de cruce aleatorio
        punto_cruce = random.randint(1, self.planificador.n_dias - 1)
        
        # Intercambiar bloques de días
        hijo.matriz[punto_cruce:] = padre2.matriz[punto_cruce:]
        
        return hijo
    
    def mutacion_hibrida(self, cromosoma: Cromosoma) -> Cromosoma:
        """Mutación híbrida: suave + búsqueda local"""
        mutado = cromosoma.copiar()
        
        # Paso 1: Mutación suave (5% de celdas disponibles)
        celdas_disponibles = []
        for j in range(self.planificador.n_dias):
            for k in range(self.planificador.n_turnos):
                if self.planificador.disponibilidad[j, k] == 1:
                    celdas_disponibles.append((j, k))
        
        n_mutaciones = max(1, int(0.05 * len(celdas_disponibles)))
        celdas_mutacion = random.sample(celdas_disponibles, n_mutaciones)
        
        for j, k in celdas_mutacion:
            # Asignar nueva asignatura aleatoria con probabilidad proporcional a prioridad
            probs = []
            for i in range(self.planificador.M):
                p_i = self.planificador.prioridad[i, j]
                probs.append(p_i)
            
            if sum(probs) > 0:
                probs = np.array(probs) / sum(probs)
                nueva_asignatura = np.random.choice(self.planificador.M, p=probs) + 1
                mutado.matriz[j, k] = nueva_asignatura
        
        # Paso 2: Búsqueda local 2-opt en días adyacentes
        self._busqueda_local_2opt(mutado)
        
        return mutado
    
    def _busqueda_local_2opt(self, cromosoma: Cromosoma):
        """Búsqueda local 2-opt intercambiando asignaturas en días adyacentes"""
        mejor_fitness = self.evaluar_fitness(cromosoma)
        
        for j in range(self.planificador.n_dias - 1):
            for k1 in range(self.planificador.n_turnos):
                for k2 in range(self.planificador.n_turnos):
                    if (self.planificador.disponibilidad[j, k1] == 1 and 
                        self.planificador.disponibilidad[j+1, k2] == 1):
                        
                        # Intercambiar asignaturas
                        temp = cromosoma.matriz[j, k1]
                        cromosoma.matriz[j, k1] = cromosoma.matriz[j+1, k2]
                        cromosoma.matriz[j+1, k2] = temp
                        
                        # Evaluar nuevo fitness
                        nuevo_fitness = self.evaluar_fitness(cromosoma)
                        
                        if nuevo_fitness > mejor_fitness:
                            mejor_fitness = nuevo_fitness
                        else:
                            # Revertir intercambio
                            cromosoma.matriz[j+1, k2] = cromosoma.matriz[j, k1]
                            cromosoma.matriz[j, k1] = temp
    
    def ejecutar(self) -> Cromosoma:
        """Ejecutar algoritmo genético completo"""
        # Inicializar población
        self.inicializar_poblacion()
        
        # Evaluar población inicial
        for cromosoma in self.poblacion:
            self.evaluar_fitness(cromosoma)
        
        # Evolución
        for generacion in range(self.planificador.params.generaciones):
            # Ordenar población por fitness
            self.poblacion.sort(key=lambda x: x.fitness, reverse=True)
            
            # Guardar mejor solución
            if self.mejor_solucion is None or self.poblacion[0].fitness > self.mejor_solucion.fitness:
                self.mejor_solucion = self.poblacion[0].copiar()
            
            # Guardar historial
            self.historial_fitness.append(self.poblacion[0].fitness)
            
            # Crear nueva población
            nueva_poblacion = []
            
            # Élite (10% mejores)
            n_elite = int(self.planificador.params.elite_porcentaje * 
                         self.planificador.params.tamaño_poblacion)
            nueva_poblacion.extend([c.copiar() for c in self.poblacion[:n_elite]])
            
            # Generar resto de la población
            while len(nueva_poblacion) < self.planificador.params.tamaño_poblacion:
                # Selección
                padre1 = self.seleccion_torneo()
                padre2 = self.seleccion_torneo()
                
                # Cruce
                if random.random() < self.planificador.params.prob_cruce:
                    hijo = self.cruce_bloques_dias(padre1, padre2)
                else:
                    hijo = padre1.copiar()
                
                # Mutación
                if random.random() < self.planificador.params.prob_mutacion:
                    hijo = self.mutacion_hibrida(hijo)
                
                # Evaluar y agregar
                self.evaluar_fitness(hijo)
                nueva_poblacion.append(hijo)
            
            self.poblacion = nueva_poblacion
            
            # Progreso cada 50 generaciones
            if generacion % 50 == 0:
                print(f"Generación {generacion}: Mejor fitness = {self.mejor_solucion.fitness:.6f}")
        
        return self.mejor_solucion

class EvaluadorPlanes:
    """Sistema de evaluación y comparación de planes de estudio"""
    
    def __init__(self, planificador: PlanificadorEstudios):
        self.planificador = planificador
    
    def _calcular_metricas_cromosoma(self, cromosoma: Cromosoma) -> Tuple[float, float, float]:
        """Calcular métricas S, P, C para un cromosoma"""
        # Métrica S: Espaciamiento óptimo
        S = self._calcular_espaciamiento(cromosoma)
        
        # Métrica P: Preparación para exámenes
        P = self._calcular_preparacion_examenes(cromosoma)
        
        # Métrica C: Balance de carga diaria
        C = self._calcular_balance_carga(cromosoma)
        
        return S, P, C
    
    def _calcular_espaciamiento(self, cromosoma: Cromosoma) -> float:
        """Calcular métrica de espaciamiento óptimo"""
        S_total = 0.0
        
        for i in range(self.planificador.M):
            # Encontrar días donde se estudia la asignatura i
            dias_estudio = []
            for j in range(self.planificador.n_dias):
                if np.any(cromosoma.matriz[j, :] == i + 1):
                    dias_estudio.append(j)
            
            if len(dias_estudio) > 1:
                # Calcular intervalos entre días consecutivos
                intervalos = []
                for idx in range(len(dias_estudio) - 1):
                    intervalo = dias_estudio[idx + 1] - dias_estudio[idx]
                    intervalos.append(intervalo)
                
                # Calcular varianza de intervalos
                if len(intervalos) > 0:
                    varianza = np.var(intervalos)
                    S_i = -varianza  # Valores altos indican espaciamiento uniforme
                else:
                    S_i = 0
            else:
                S_i = 0  # No hay suficientes sesiones para calcular espaciamiento
            
            # Ponderar por dificultad
            S_total += self.planificador.dificultad[i] * S_i
        
        return S_total / self.planificador.M
    
    def _calcular_preparacion_examenes(self, cromosoma: Cromosoma) -> float:
        """Calcular métrica de preparación para exámenes"""
        P_total = 0.0
        asignaturas_con_examenes = 0
        
        for i in range(self.planificador.M):
            recencias = []
            
            # Para cada día de examen de la asignatura i
            for j_prima in range(self.planificador.n_dias + 1):
                if self.planificador.calendario_examenes[j_prima, i] == 1:
                    # Encontrar último día de estudio antes del examen
                    ultimo_dia_estudio = -1
                    for j in range(min(j_prima, self.planificador.n_dias)):
                        if np.any(cromosoma.matriz[j, :] == i + 1):
                            ultimo_dia_estudio = j
                    
                    if ultimo_dia_estudio >= 0:
                        recencia = 1.0 / (j_prima - ultimo_dia_estudio)
                        recencias.append(recencia)
                    else:
                        recencias.append(0)  # No se estudió antes del examen
            
            if recencias:
                P_i = np.mean(recencias)
                asignaturas_con_examenes += 1
            else:
                P_i = 0
            
            # Ponderar por dificultad
            P_total += self.planificador.dificultad[i] * P_i
        
        return P_total / max(1, asignaturas_con_examenes)
    
    def _calcular_balance_carga(self, cromosoma: Cromosoma) -> float:
        """Calcular métrica de balance de carga diaria"""
        penalty_total = 0.0
        umbral = self.planificador.params.umbral_carga * self.planificador.n_turnos
        
        for j in range(self.planificador.n_dias):
            # Calcular carga del día j
            carga_dia = np.sum(cromosoma.matriz[j, :] > 0)
            
            # Penalizar si excede umbral
            if carga_dia > umbral:
                penalty_total += (carga_dia - umbral)
        
        return -penalty_total  # Valores altos indican mejor balance
    
    def evaluar_plan(self, cromosoma: Cromosoma) -> float:
        """Evaluar un plan usando la función de evaluación global"""
        S, P, C = self._calcular_metricas_cromosoma(cromosoma)
        
        F = (self.planificador.params.w_S * S + 
             self.planificador.params.w_P * P + 
             self.planificador.params.w_C * C)
        
        return F
    
    def comparar_planes(self, plan_A: Cromosoma, plan_B: Cromosoma) -> str:
        """Comparar dos planes y determinar cuál es mejor"""
        F_A = self.evaluar_plan(plan_A)
        F_B = self.evaluar_plan(plan_B)
        
        diferencia = abs(F_A - F_B)
        
        if diferencia < self.planificador.params.epsilon:
            return f"Los planes son equivalentes (diferencia: {diferencia:.6f})"
        elif F_A > F_B:
            return f"Plan A es mejor (F_A: {F_A:.6f} > F_B: {F_B:.6f})"
        else:
            return f"Plan B es mejor (F_B: {F_B:.6f} > F_A: {F_A:.6f})"

def ejemplo_uso():
    """Ejemplo de uso del sistema completo"""
    print("=== Sistema de Planificación de Estudios ===\n")
    
    # Configurar problema
    n_dias = 14  # 2 semanas
    n_turnos = 4  # 4 turnos por día
    asignaturas = ["Matemáticas", "Física", "Química", "Historia", "Literatura"]
    
    # Crear planificador
    params = Parametros(
        tamaño_poblacion=50,
        generaciones=200,
        w_S=0.3, w_P=0.5, w_C=0.2
    )
    
    planificador = PlanificadorEstudios(n_dias, n_turnos, asignaturas, params)
    
    # Configurar parámetros del problema
    # Sesiones mínimas requeridas
    sesiones_minimas = np.array([4, 3, 3, 2, 2])
    planificador.configurar_sesiones_minimas(sesiones_minimas)
    
    # Dificultades (1-10)
    dificultades = np.array([8, 7, 6, 4, 5])
    planificador.configurar_dificultades(dificultades)
    
    # Disponibilidad (estudiante no disponible algunos turnos)
    disponibilidad = np.ones((n_dias, n_turnos))
    # Simular que no está disponible en algunos turnos
    disponibilidad[0, :2] = 0  # Primer día, primeros 2 turnos no disponible
    disponibilidad[6:8, 3] = 0  # Fines de semana, último turno no disponible
    planificador.configurar_disponibilidad(disponibilidad)
    
    # Configurar calendario de exámenes
    examenes = np.zeros((n_dias + 1, len(asignaturas)))
    examenes[n_dias, 0] = 1  # Examen de Matemáticas al final
    examenes[n_dias, 1] = 1  # Examen de Física al final
    examenes[10, 2] = 1      # Examen de Química en día 10
    planificador.configurar_examenes(examenes)
    
    # Ejecutar algoritmo genético
    print("Ejecutando algoritmo genético...")
    ag = AlgoritmoGenetico(planificador)
    mejor_plan = ag.ejecutar()
    
    print(f"\n=== Resultados ===")
    print(f"Mejor fitness obtenido: {mejor_plan.fitness:.6f}")
    
    # Evaluar plan con métricas detalladas
    evaluador = EvaluadorPlanes(planificador)
    S, P, C = evaluador._calcular_metricas_cromosoma(mejor_plan)
    F = evaluador.evaluar_plan(mejor_plan)
    
    print(f"Espaciamiento (S): {S:.6f}")
    print(f"Preparación exámenes (P): {P:.6f}")
    print(f"Balance carga (C): {C:.6f}")
    print(f"Evaluación global (F): {F:.6f}")
    
    # Mostrar plan generado
    print(f"\n=== Plan de Estudios Generado ===")
    for j in range(n_dias):
        print(f"Día {j+1:2d}: ", end="")
        for k in range(n_turnos):
            if mejor_plan.matriz[j, k] > 0:
                asig_idx = mejor_plan.matriz[j, k] - 1
                print(f"{asignaturas[asig_idx][:4]:>4} ", end="")
            else:
                print("---- ", end="")
        print()
    
    # Generar segundo plan para comparación
    print(f"\nGenerando segundo plan para comparación...")
    ag2 = AlgoritmoGenetico(planificador)
    segundo_plan = ag2.ejecutar()
    
    # Comparar planes
    resultado_comparacion = evaluador.comparar_planes(mejor_plan, segundo_plan)
    print(f"\n=== Comparación de Planes ===")
    print(resultado_comparacion)
    
    return planificador, mejor_plan, segundo_plan

if __name__ == "__main__":
    planificador, plan_A, plan_B = ejemplo_uso()