from collections import deque
class Graph:
    def __init__(self, V):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def addEdge(self, v, w):
        self.adj[v].append(w)

    def BFS(self, s):
        visited = [False] * self.V
        queue = deque()
        visited[s] = True
        queue.append(s)


        while queue:
            s = queue.popleft()
            print (s,end=' ')

            for i in self.adj[s]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

################################### DFS FUNCTIONS ###############################
    def DFS_recurr(self, v, visited):
        visited[v] = True
        print(v, end=" ")

        for i in self.adj[v]:
            if not visited[i]:
                self.DFS_recurr(i, visited)

    def DFS(self, s):
        visited = [False] * self.V
        self.DFS_recurr(s, visited)

def main():
    g = Graph(6)
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 3)
    g.addEdge(2, 4)
    g.addEdge(2, 5)

    print("Breadth First Traversal: ", end="")
    g.BFS(0)
    print()
    print("Depth First Traversal: ", end="")
    g.DFS(0)
main()