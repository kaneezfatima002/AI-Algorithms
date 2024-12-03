class PuzzleNode:
    def __init__(self, state, parent=None, move=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def generate_children(self):
        children = []
        empty_pos = self.state.index(0)
        row, col = divmod(empty_pos, 3)
        moves = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}

        for move, (dr, dc) in moves.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_pos = new_row * 3 + new_col
                new_state = list(self.state)
                new_state[empty_pos], new_state[new_pos] = new_state[new_pos], new_state[empty_pos]
                children.append(PuzzleNode(tuple(new_state), self, move, self.g_cost + 1))
        return children

    def calculate_heuristic(self, goal_state):
        h = 0
        for i in range(1, 9):
            curr_pos = self.state.index(i)
            goal_pos = goal_state.index(i)
            curr_row, curr_col = divmod(curr_pos, 3)
            goal_row, goal_col = divmod(goal_pos, 3)
            h += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return h


class AStarSolver:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

    def solve(self):
        open_list = [PuzzleNode(self.start_state)]
        closed_list = set()

        while open_list:
            open_list.sort(key=lambda x: x.f_cost)
            current_node = open_list.pop(0)

            if current_node.state == self.goal_state:
                return self.get_solution_path(current_node)
            closed_list.add(current_node.state)

            for child in current_node.generate_children():
                if child.state in closed_list:
                    continue
                child.h_cost = child.calculate_heuristic(self.goal_state)
                child.f_cost = child.g_cost + child.h_cost
                open_list.append(child)

        return None

    def get_solution_path(self, node):
        path = []
        while node.parent:
            path.append(node.move)
            node = node.parent
        return path[::-1]

start_state = (1, 2, 3, 5, 8, 0, 4, 6, 7)
goal_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)

solver = AStarSolver(start_state, goal_state)
solution = solver.solve()

if solution:
    print("Solution found!")
    print("Moves:", solution)
else:
    print("No solution founf")
