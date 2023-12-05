# Constants for the game
M = 6  # Number of rows
N = 7  # Number of columns
IN_A_ROW = 4  # Number of pieces in a row needed to win


def get_blank_board():
    return [[0 for _ in range(M)] for _ in range(N)]


def in_a_row_met(arr: list):
    streakR = 0
    streakY = 0
    for p in arr:
        if p == -1:
            streakY = 0
            streakR += 1
            if streakR == IN_A_ROW:
                return -1  # Red wins
        elif p == 1:
            streakR = 0
            streakY += 1
            if streakY == IN_A_ROW:
                return 1  # Yellow wins
        else:
            streakR, streakY = 0, 0
    return 0  # No one wins


def print_board(board):
    rows, cols = M, N  # Dimensions of the board
    col_labels = "ABCDEFG"  # Column labels

    # Mapping of values to display characters
    display_mapping = {
        -1: "R",  # Red piece
        0: " ",  # Empty space
        1: "Y",  # Yellow piece
    }

    # Print column labels
    print("\n  " + "   ".join(col_labels))

    # Print the top border of the board
    print("+" + "---+" * cols)

    # Print each row of the board
    for row in range(rows):
        # Print each cell in the row
        for col in range(cols):
            print("| " + display_mapping[board[col][row]] + " ", end="")
        print("|")  # End of row
        print("+" + "---+" * cols)  # Bottom border of each cell

    print()


def insert_piece_optimized(board: list, col_num: int, player: int):
    col = board[col_num]
    row_idx = None
    for i in range(M - 1, -1, -1):
        if col[i] == 0:
            col[i] = player
            row_idx = i
            break
    board[col_num] = col
    return board, col_num, row_idx


def game_over_optimized(board, col_idx, row_idx, player):
    """
    Checks if the game is over after a piece is inserted.
    Only checks the row, column, and diagonals affected by the last move.
    """
    # Check the row
    if in_a_row_met(board[col_idx]):
        return player

    # Check the column
    col = [board[i][row_idx] for i in range(N)]
    if in_a_row_met(col):
        return player

    # Check diagonals
    fdiag, bdiag = [], []
    for i in range(N):
        for j in range(M):
            if i + j == col_idx + row_idx:
                fdiag.append(board[i][j])
            if i - j == col_idx - row_idx:
                bdiag.append(board[i][j])

    if in_a_row_met(fdiag) or in_a_row_met(bdiag):
        return player

    return 0  # Game continues


def make_move(board: list, col_num: int, player: int):
    """
    Makes a move for the given player in the specified column.
    Returns the updated board and the game status (-1, 0, or 1).
    """
    # Insert the piece and get the row index where it was placed
    board, col_num, row_idx = insert_piece_optimized(board, col_num, player)

    # Check if this move resulted in a game over
    status = game_over_optimized(board, col_num, row_idx, player)

    return board, status


def get_legal_moves(board):
    legal_moves = []
    for col_index in range(N):  # Iterate through each column
        if board[col_index][0] == 0:  # Check if the top cell of the column is empty
            legal_moves.append(col_index)  # If empty, it's a legal move
    return legal_moves


def evaluate_board(board):
    # Check for win/loss
    for col in range(N):
        for row in range(M):
            if board[col][row] != 0:
                # Check if the current player has a winning move
                if game_over_optimized(board, col, row, board[col][row]):
                    return 100 if board[col][row] == 1 else -100
    return 0


def minimax(board, depth, is_maximizing_player, alpha, beta):
    legal_moves = get_legal_moves(board)
    game_result = evaluate_board(board)

    # Base case: game over or depth limit reached
    if game_result != 0 or depth == 0 or not legal_moves:
        return game_result

    if is_maximizing_player:
        max_eval = float("-inf")
        for col in legal_moves:
            new_board, _, row_idx = insert_piece_optimized(
                board.copy(), col, 1
            )  # AI is 1
            eval = minimax(new_board, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for col in legal_moves:
            new_board, _, row_idx = insert_piece_optimized(
                board.copy(), col, -1
            )  # Opponent is -1
            eval = minimax(new_board, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def find_best_move(board, depth, ai_player):
    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for col in get_legal_moves(board):
        temp_board = [row[:] for row in board]  # Create a copy of the board
        temp_board, _, _ = insert_piece_optimized(temp_board, col, ai_player)
        move_eval = minimax(temp_board, depth - 1, False, alpha, beta)

        if move_eval > best_score:
            best_score = move_eval
            best_move = col

    return best_move


def human_vs_ai():
    board = get_blank_board()
    game_over = False
    ai_player = 1  # Let's assume AI plays as '1' (Yellow)
    human_player = -1  # Human plays as '-1' (Red)

    while not game_over:
        # Print current board
        print_board(board)

        # Human's Turn
        human_move = int(
            input("Your move (0-6): ")
        )  # Assuming 0 to 6 are valid column inputs
        board, _, _ = insert_piece_optimized(board, human_move, human_player)
        game_status = evaluate_board(board)
        if game_status != 0:
            print(
                "Congratulations! You've won!"
                if game_status == -100
                else "It's a draw!"
            )
            break

        # AI's Turn
        print("AI is making its move...")
        ai_move = find_best_move(board, 1000, ai_player)  # Adjust depth as needed
        board, _, _ = insert_piece_optimized(board, ai_move, ai_player)
        print_board(board)
        game_status = evaluate_board(board)
        if game_status != 0:
            print("AI wins!" if game_status == 100 else "It's a draw!")
            break

    # Print the final board state
    print_board(board)


human_vs_ai()
# # Streamlined Example Usage
# board = get_blank_board()

# # Making moves and checking for game status
# moves = [
#     (0, 1),
#     (1, -1),
#     (2, 1),
#     (3, -1),
#     (3, 1),
#     (2, -1),
#     (1, 1),
#     (1, -1),
#     (0, 1),
#     (0, 1),
#     (0, 1),
# ]

# for col, player in moves:
#     board, status = make_move(board, col, player)
#     print_board(board)
#     print(get_legal_moves(board))
#     if status != 0:
#         print(f"Player {player} wins!")
#         break

# # Print the final board state for visualization
# print_board(board)
