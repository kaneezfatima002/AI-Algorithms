class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    def dls(self, current, goal, limit):
        if current == goal:
            return True
        if limit <= 0:
            return False
        for neighbor in self.graph.get(current, []):
            if self.dls(neighbor, goal, limit - 1):
                return True
        return False
    def iddfs(self, start, goal, max_depth):
        for depth in range(max_depth + 1):
            if self.dls(start, goal, depth):
                return True
        return False


def main():
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)

    start = 0
    goal = 9
    max_depth = 3

    if g.iddfs(start, goal, max_depth):
        print(f"Goal {goal} found!")
    else:
        print(f"Goal {goal} not found within depth {max_depth}.")

main()