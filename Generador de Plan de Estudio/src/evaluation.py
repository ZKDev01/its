
from .models import *

WEIGHTS = {'alpha': 0.3, 'beta': 0.2, 'gamma': 0.4, 'delta': 0.1}

def calculate_fitness(plan, student, graph):
    U_con, U_pref, U_obj, P_dif = 0, 0, 0, 0
    for topic, subtopics in plan.assignment.items():
        if not subtopics:
            continue
        k_i = student.knowledge.get(topic, sum(student.knowledge.get(s, 0) for s in subtopics) / len(subtopics))
        U_con += (1 - k_i)
        U_pref += student.pref_norm.get(topic, 0.0)
        U_obj += 1.0 if topic in student.goal_required_topics() else 0.0
        P_dif += student.diff_norm.get(topic, 0.5) * (1 - k_i)
    return (WEIGHTS['alpha'] * U_con +
            WEIGHTS['beta'] * U_pref +
            WEIGHTS['gamma'] * U_obj -
            WEIGHTS['delta'] * P_dif)

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
