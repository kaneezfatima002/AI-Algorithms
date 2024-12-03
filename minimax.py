class Minimax:
    def __init__(self, board):
        self.board = board

    def is_terminal(self, state):
        for row in range(3):
            if state[row][0] == state[row][1] == state[row][2] != 0:
                return True
        for col in range(3):
            if state[0][col] == state[1][col] == state[2][col] != 0:
                return True
        if state[0][0] == state[1][1] == state[2][2] != 0:
            return True
        if state[0][2] == state[1][1] == state[2][0] != 0:
            return True
        return not any(0 in row for row in state)

    def utility(self, state):
        for row in range(3):
            if state[row][0] == state[row][1] == state[row][2] == 1:
                return 1
            if state[row][0] == state[row][1] == state[row][2] == -1:
                return -1
        for col in range(3):
            if state[0][col] == state[1][col] == state[2][col] == 1:
                return 1
            if state[0][col] == state[1][col] == state[2][col] == -1:
                return -1
        if state[0][0] == state[1][1] == state[2][2] == 1:
            return 1
        if state[0][2] == state[1][1] == state[2][0] == 1:
            return 1
        if state[0][0] == state[1][1] == state[2][2] == -1:
            return -1
        if state[0][2] == state[1][1] == state[2][0] == -1:
            return -1
        return 0

    def minimax(self, state, depth, maximizing_player):
        if self.is_terminal(state):
            return self.utility(state)

        if maximizing_player:
            max_eval = float('-inf')
            for row in range(3):
                for col in range(3):
                    if state[row][col] == 0:
                        state[row][col] = 1
                        eval = self.minimax(state, depth + 1, False)
                        state[row][col] = 0
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(3):
                for col in range(3):
                    if state[row][col] == 0:
                        state[row][col] = -1
                        eval = self.minimax(state, depth + 1, True)
                        state[row][col] = 0
                        min_eval = min(min_eval, eval)
            return min_eval

    def best_move(self):
        best_val = float('-inf')
        best_move = (-1, -1)
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:
                    self.board[row][col] = 1
                    move_val = self.minimax(self.board, 0, False)
                    self.board[row][col] = 0
                    if move_val > best_val:
                        best_val = move_val
                        best_move = (row, col)
        return best_move

    def print_board(self):
        for row in self.board:
            print(" | ".join(str(x) if x != 0 else " " for x in row))
            print("-" * 5)


def main():
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    minimax = Minimax(board)

    while True:
        minimax.print_board()

        x, y = map(int, input("Enter your move (row and column): ").split())
        if board[x][y] != 0:
            print("Invalid move. Try again.")
            continue
        board[x][y] = 1

        if minimax.is_terminal(board):
            minimax.print_board()
            print("X wins!" if minimax.utility(board) == 1 else "It's a draw!")
            break

        print("AI is making a move...")
        ai_move = minimax.best_move()
        board[ai_move[0]][ai_move[1]] = -1

        if minimax.is_terminal(board):
            minimax.print_board()
            print("O wins!" if minimax.utility(board) == -1 else "It's a draw!")
            break


if __name__ == "__main__":
    main()
