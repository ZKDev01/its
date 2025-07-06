
import json

class StudentProfile:
    def __init__(self, knowledge, preferences, difficulty, goal, total_time=None, objectives_file="../data/objectives.json"):
        self.knowledge = knowledge
        self.preferences = preferences
        self.difficulty = difficulty
        self.goal = goal
        self.total_time = total_time  # Se incluye total_time correctamente

        # Cargar los objetivos desde un JSON externo
        with open(objectives_file, "r") as f:
            self.objectives_map = json.load(f)

        # Normalización de preferencias y dificultad
        max_pref = max(preferences.values(), default=1)
        diff_map = {1: 0.2, 2: 0.2, 3: 0.5, 4: 0.5, 5: 0.8}
        self.pref_norm = {k: v / max_pref for k, v in preferences.items()}
        self.diff_norm = {k: diff_map.get(difficulty[k], 0.8) for k in difficulty}

    def goal_required_topics(self):
        return set(self.objectives_map.get(self.goal, []))


class KnowledgeGraph:
    def __init__(self, nodes):
        self.nodes = nodes
        self.prereq_map = {k: v.get("prerequisites", []) for k, v in nodes.items()}
        self.topic_map = {}
        for sub, meta in nodes.items():
            topic = meta["topic"]
            if topic not in self.topic_map:
                self.topic_map[topic] = []
            self.topic_map[topic].append(sub)

    def prerequisites(self, sub):
        return self.prereq_map.get(sub, [])

    def is_prerequisite_satisfied(self, sub, student, plan, threshold=0.6):
        for prereq in self.prerequisites(sub):
            if student.knowledge.get(prereq, 0) >= threshold or prereq in plan.all_subtopics():
                continue
            return False
        return True

    def get_subtopics(self, topic):
        return self.topic_map.get(topic, [])

    def all_topics(self):
        return list(self.topic_map.keys())


class StudyPlan:
    def __init__(self, assignment: dict):
        self.assignment = assignment  # e.g., {"Algoritmos": ["Prog. Dinámica", "Grafos", ...]}

    def topics(self):
        return list(self.assignment.keys())

    def subtopics(self, topic):
        return self.assignment.get(topic, [])

    def all_subtopics(self):
        return [s for subs in self.assignment.values() for s in subs]
