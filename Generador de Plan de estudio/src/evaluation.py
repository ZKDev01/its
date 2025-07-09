
from .models import *

WEIGHTS = {'alpha': 0.3, 'beta': 0.2, 'gamma': 0.4, 'delta': 0.1}

def calculate_fitness(plan, student, graph, weights=None):
    alpha = weights.get("alpha", 0.3) if weights else 0.3
    beta = weights.get("beta", 0.2) if weights else 0.2
    gamma = weights.get("gamma", 0.4) if weights else 0.4
    delta = weights.get("delta", 0.1) if weights else 0.1

    # U_con (conocimiento nuevo)
    subtopics = plan.all_subtopics()
    knowledge_sum = sum(1 - student.knowledge.get(s, 0) for s in subtopics)
    U_con = knowledge_sum / len(subtopics) if subtopics else 0

    # U_pref (preferencias del estudiante por tema)
    pref_sum = sum(student.pref_norm.get(t, 0) for t in plan.topics())
    U_pref = pref_sum / len(plan.topics()) if plan.topics() else 0

    # U_obj (temas que pertenecen al objetivo)
    required = student.goal_required_topics()
    if required:
        count = sum(1 for t in plan.topics() if t in required)
        U_obj = count / len(required)
    else:
        U_obj = 0

    # P_dif (penalización por dificultad)
    P_dif = 0
    for t in plan.topics():
        for s in plan.subtopics(t):
            P_dif += student.diff_norm.get(s, 0.5)
    P_dif = P_dif / len(subtopics) if subtopics else 0

    # Fitness total
    return alpha * U_con + beta * U_pref + gamma * U_obj - delta * P_dif


def is_valid_plan(plan, student, graph, n, m):
    if len(plan.assignment) < n:
        return False
    for topic, subs in plan.assignment.items():
        if len(subs) < m:
            return False
        for sub in subs:
            if not graph.is_prerequisite_satisfied(sub, student, plan):
                return False
    for req in student.goal_required_topics():
        if req not in plan.assignment:
            return False
    return True
