class Node:
    def __init__(self, state, parent, move, h_cost):
        self.state = state
        self.parent = parent
        self.move = move
        self.h_cost = h_cost

    def generate_children(self, goal_state):
        children = []
        empty_index = self.state.index(0)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        empty_row, empty_col = divmod(empty_index, 3)

        for move in moves:
            new_row = empty_row + move[0]
            new_col = empty_col + move[1]

            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = self.state[:]
                new_index = new_row * 3 + new_col
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
                h_cost = self.calculate_heuristic(new_state, goal_state)
                children.append(Node(new_state, self, move, h_cost))
        return children

    def calculate_heuristic(self, state, goal_state):
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0:
                goal_index = goal_state.index(tile)
                current_row, current_col = divmod(i, 3)
                goal_row, goal_col = divmod(goal_index, 3)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance


class GreedyBestFirstSearch:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state

    def solve(self):
        open_list = [Node(self.start_state, None, None, self.calculate_heuristic(self.start_state))]
        closed_list = []

        while open_list:
            current_node = open_list.pop(0)

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

            closed_list.append(current_node)

            for child in current_node.generate_children(self.goal_state):
                if child.state not in [node.state for node in closed_list]:
                    open_list.append(child)

            open_list.sort(key=lambda node: node.h_cost)

        return None

    def calculate_heuristic(self, state):
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0:
                goal_index = self.goal_state.index(tile)
                current_row, current_col = divmod(i, 3)
                goal_row, goal_col = divmod(goal_index, 3)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    def trace_solution(self, node):
        path = []
        while node.parent is not None:
            path.append(node.move)
            node = node.parent
        return path[::-1]


# Example usage:
start_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

gbfs = GreedyBestFirstSearch(start_state, goal_state)
solution = gbfs.solve()

if solution:
    print("Moves to solve the puzzle:", solution)
else:
    print("No solution found.")
