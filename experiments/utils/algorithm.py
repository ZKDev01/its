
import random
from .evaluation import calculate_fitness, is_valid_plan
from .models import StudyPlan

def generate_random_plan(graph, student, n, m):
    plan = {}
    topics = random.sample(graph.all_topics(), len(graph.all_topics()))
    selected = 0
    for t in topics:
        subs = [s for s in graph.get_subtopics(t)
                if student.knowledge.get(s, 0) < 0.4 and graph.is_prerequisite_satisfied(s, student, StudyPlan(plan))]
        if len(subs) >= m:
            plan[t] = random.sample(subs, m)
            selected += 1
            if selected >= n:
                break
    return StudyPlan(plan)

def crossover(p1, p2):
    keys1 = list(p1.assignment.keys())
    cut = len(keys1) // 2
    child1_data = {k: p1.assignment[k] for k in keys1[:cut]}
    for k in p2.assignment:
        if k not in child1_data:
            child1_data[k] = p2.assignment[k]
    return StudyPlan(child1_data)

def mutate(plan, graph, student, m):
    if not plan.assignment:
        return plan
    topic = random.choice(list(plan.assignment.keys()))
    subs = [s for s in graph.get_subtopics(topic)
            if student.knowledge.get(s, 0) < 0.4 and s not in plan.assignment[topic]]
    if subs:
        plan.assignment[topic][random.randint(0, m-1)] = random.choice(subs)
    return plan

def genetic_optimize(student, graph, n, m, pop_size=30, generations=50, weights=None):
    from .evaluation import calculate_fitness, is_valid_plan
    from .models import StudyPlan
    import random

    population = [generate_random_plan(graph, student, n, m) for _ in range(pop_size)]
    best_plan = None
    best_fitness = float('-inf')

    for gen in range(generations):
        scored = [
            (p, calculate_fitness(p, student, graph, weights))
            for p in population if is_valid_plan(p, student, graph, n, m)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"[Gen {gen}] Planes válidos: {len(scored)}")

        if len(scored) == 0:
            print("⚠ No se encontraron planes válidos.")
            break

        if scored[0][1] > best_fitness:
            best_fitness = scored[0][1]
            best_plan = scored[0][0]

        parents = [x[0] for x in scored]

        if len(parents) < 2:
            parents = parents * 2

        # ✅ Elitismo
        children = [scored[0][0]]

        while len(children) < pop_size:
            i1 = random.randint(0, len(parents) - 1)
            i2 = random.randint(0, len(parents) - 1)
            child = crossover(parents[i1], parents[i2])
            if random.random() < 0.3:
                child = mutate(child, graph, student, m)
            children.append(child)

        population = children

    return best_plan, best_fitness
