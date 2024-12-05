import math
PLAYER = 'X'
AI = 'O'
EMPTY = '_'

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def check_winner(board):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != EMPTY:
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != EMPTY:
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]

    for row in board:
        if EMPTY in row:
            return None
    return 'Tie'

def is_moves_left(board):
    for row in board:
        if EMPTY in row:
            return True
    return False

def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == AI:
        return 10 - depth
    elif winner == PLAYER:
        return depth - 10
    elif winner == 'Tie':
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = AI
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = EMPTY
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = PLAYER
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = EMPTY
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

def find_best_move(board):
    best_value = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                board[i][j] = AI
                move_value = minimax(board, 0, False, -math.inf, math.inf)
                board[i][j] = EMPTY

                if move_value > best_value:
                    best_value = move_value
                    best_move = (i, j)

    return best_move

def play_game():
    board = [[EMPTY for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe! You are 'X' and AI is 'O'.")
    print_board(board)

    while True:
        row, col = map(int, input("Enter your move (row and column): ").split())
        if board[row][col] != EMPTY:
            print("Invalid move! Try again.")
            continue
        board[row][col] = PLAYER
        print_board(board)

        if check_winner(board):
            break

        print("AI is making a move...")
        ai_move = find_best_move(board)
        board[ai_move[0]][ai_move[1]] = AI
        print_board(board)

        if check_winner(board):
            break

    result = check_winner(board)
    if result == 'Tie':
        print("It's a tie!")
    else:
        print(f"{result} wins!")

if __name__ == "__main__":
    play_game()
