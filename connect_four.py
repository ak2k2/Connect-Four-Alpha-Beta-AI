import copy
import math
import random

# Constants for the game
M = 7  # Number of rows
N = 8  # Number of columns
IN_A_ROW = 4  # Number of pieces in a row needed to win

# Constants for players
AI_PLAYER = 1  # Representing the AI
HUMAN_PLAYER = -1  # Representing the human

"""
RED is -1
YELLOW is +1
"""


def get_blank_board():
    return [[0 for _ in range(M)] for _ in range(N)]


def in_a_row_met(arr: list, in_a_row=IN_A_ROW):
    """
    if in_a_row=IN_A_ROW streaks mark game over conditions.
    if in_a_row < IN_A_ROW then steraks can be used to count sequences for evaluation.
    """
    streakR = 0
    streakY = 0
    for p in arr:
        if p == -1:
            streakY = 0
            streakR += 1
            if streakR == in_a_row:
                return -1  # Red wins
        elif p == 1:
            streakR = 0
            streakY += 1
            if streakY == in_a_row:
                return 1  # Yellow wins
        else:
            streakR, streakY = 0, 0
    return 0  # No one wins


def print_board(board):
    rows, cols = M, N  # Dimensions of the board
    files = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    col_labels = files[:cols]  # Labels for the columns

    # ANSI color codes
    ANSI_RED = "\033[91m"
    ANSI_YELLOW = "\033[93m"
    ANSI_RESET = "\033[0m"
    ANSI_WHITE = "\033[97m"

    # Mapping of values to display characters with colors
    display_mapping = {
        -1: ANSI_RED + "R" + ANSI_RESET,  # Red piece
        0: " ",  # Empty space
        1: ANSI_YELLOW + "Y" + ANSI_RESET,  # Yellow piece
    }

    # Print column labels
    print("\n  " + "   ".join(col_labels))

    # Print the top border of the board
    print("+" + "---+" * cols)

    # Print each row of the board
    for row in range(rows):
        # Print each cell in the row
        for col in range(cols):
            cell_value = display_mapping[board[col][row]]
            print(f"| {cell_value} ", end="")
        print("|")  # End of row
        print("+" + "---+" * cols)  # Bottom border of each cell


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
    # num total pieces
    num_total_pieces = sum(1 for col in board for row in col if row != 0)

    if num_total_pieces < 7:  # no need to check for game over
        return 0

    # Actually column since were in column major order
    if in_a_row_met(board[col_idx]):
        return player

    # Actually row since were in column major order
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


def brute_force_game_over(board):
    num_total_pieces = sum(1 for row in board for cell in row if cell != 0)
    if num_total_pieces < 7:  # Assuming 7 is the minimum number to check for game over
        return 0

    max_row, max_col = len(board), len(board[0])

    # Check rows and columns
    for r in range(max_row):
        for c in range(max_col - 3):
            if in_a_row_met([board[r][c + i] for i in range(4)]):
                return board[r][c]

    for c in range(max_col):
        for r in range(max_row - 3):
            if in_a_row_met([board[r + i][c] for i in range(4)]):
                return board[r][c]

    # Check diagonals
    for r in range(max_row - 3):
        for c in range(max_col - 3):
            # Check forward diagonal
            if in_a_row_met([board[r + i][c + i] for i in range(4)]):
                return board[r][c]

            # Check backward diagonal
            if in_a_row_met([board[r + 3 - i][c + i] for i in range(4)]):
                return board[r + 3][c]

    return 0


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


def make_move_minimax(board: list, col_num: int, player: int):
    # Insert the piece and get the row index where it was placed
    board, col_num, row_idx = insert_piece_optimized(board, col_num, player)

    return board, col_num, row_idx


def get_legal_moves(board):
    legal_moves = []
    for col_index in range(N):  # Iterate through each column
        if board[col_index][0] == 0:  # Check if the top cell of the column is empty
            legal_moves.append(col_index)  # If empty, it's a legal move
    return legal_moves


def count_sequnces(arr: list, player: int, seq_len: int):
    num_seq = 0
    streak = 0
    for p in arr:
        if p == player:
            streak += 1
            if streak == seq_len:
                num_seq += 1
        else:
            streak = 0
    return num_seq


def get_diagonals(board):
    max_col = len(board[0])
    max_row = len(board)
    fdiag = [[] for _ in range(max_row + max_col - 1)]
    bdiag = [[] for _ in range(len(fdiag))]
    min_bdiag = -max_row + 1

    for x in range(max_col):
        for y in range(max_row):
            fdiag[x + y].append(board[y][x])
            bdiag[x - y - min_bdiag].append(board[y][x])

    return fdiag, bdiag


def count_sequences(seq, player, n):
    count = 0
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == [player] * n:
            count += 1
    return count


def num_seq_len_n(board, player, n):
    num_ns = 0
    # Counting for columns
    for col in board:
        num_ns += count_sequences(col, player, n)

    # Counting for rows
    for row in board:
        num_ns += count_sequences(row, player, n)

    # Get and count for diagonals
    fdiag, bdiag = get_diagonals(board)
    for diag in fdiag + bdiag:
        num_ns += count_sequences(diag, player, n)

    return num_ns


def count_center_pieces(board, player):
    center_col = N // 2
    left = center_col - 1
    right = center_col + 1

    center = sum(1 for row in board[center_col] if row == player)
    left = sum(1 for row in board[left] if row == player)
    right = sum(1 for row in board[right] if row == player)

    return (4 * center) + 3 * (left + right)


def count_shapes(board, player):
    N, M = len(board), len(board[0])
    l_count, plus_count = 0, 0

    # Check for L shape in various subgrid sizes
    def check_l_shape(subgrid):
        # Check each corner for L shape
        corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        for corner in corners:
            if all(
                subgrid[corner[0]][i] == player for i in range(len(subgrid))
            ) and all(subgrid[i][corner[1]] == player for i in range(len(subgrid))):
                return True
        return False

    # Check for plus shape in 3x3 subgrid
    def check_plus_shape(i, j):
        return all(board[i + 1][j + k] == player for k in range(3)) and all(
            board[i + k][j + 1] == player for k in range(3)
        )

    # Check for L shapes
    for size in range(2, IN_A_ROW):
        for i in range(N - size + 1):
            for j in range(M - size + 1):
                subgrid = [row[j : j + size] for row in board[i : i + size]]
                if check_l_shape(subgrid):
                    l_count += 1 * size

    # Check for plus shapes
    for i in range(N - 2):
        for j in range(M - 2):
            if check_plus_shape(i, j):
                plus_count += 1

    return int(l_count + plus_count)


def evaluate(board, player, depth_reached=0):
    status = brute_force_game_over(board)

    num_total_pieces = sum(1 for col in board for row in col if row != 0)

    if status != 0:  # the game was won
        return (
            math.inf * status
        )  # if player is max player and they won then status is 1. if they lost then status is -1.

    num_twos = num_seq_len_n(board, 1, 2) - num_seq_len_n(board, -1, 2)
    num_threes = num_seq_len_n(board, 1, 3) - num_seq_len_n(board, -1, 3)

    # bottom_center_preference = sum(
    #     1
    #     for i in range(N)
    #     for j in range(M)
    #     if board[i][j] == player
    #     and abs(i - N // 2) + abs(j - M // 2) < 2  # players bottom center pieces
    # ) - sum(
    #     1
    #     for i in range(N)
    #     for j in range(M)
    #     if board[i][j] == -player
    #     and abs(i - N // 2) + abs(j - M // 2) < 2  # opponents bottom center pieces
    # )

    center_pieces = count_center_pieces(board, 1) - count_center_pieces(board, -1)

    # shapes = count_shapes(board, 1) - count_shapes(board, -1)

    v = int(
        (50 * num_threes)
        + (30 * num_twos)
        + (0.25 * center_pieces * max(M * N - num_total_pieces, 0))
        # + (5 * bottom_center_preference * max(M * N - num_total_pieces, 0))
        # + (20 * shapes)
        + random.randint(-5, 5)
    )

    # print(f"num_twos: {num_twos}, num_threes: {num_threes}")

    if not get_legal_moves(board):  # Check for draw
        if v > 0:  # yellow is winning
            return v - 10  # avoid taking a draw when winning
        elif v < 0:  # red is winning
            return v + 10  # avoid taking a draw when winning
        else:
            return 0
    return v


def undo_move(board, col_num, row_idx):
    board[col_num][row_idx] = 0
    return board


def minimax(board, depth, alpha, beta, maximizingPlayer, depth_reached=0):
    depth_reached += 1
    game_over_status = brute_force_game_over(board)
    if (
        depth == 0
        or game_over_status != 0
        or not get_legal_moves(board)
        or not get_legal_moves(board)
    ):
        return evaluate(
            board,
            1 if maximizingPlayer else -1,
            depth_reached=depth_reached,
        )

    legal_moves = get_legal_moves(board)
    player = 1 if maximizingPlayer else 0
    legal_moves.sort(
        key=lambda x: evaluate(make_move(copy.deepcopy(board), x, player)[0], player),
        reverse=(player == 1),  # descending for max player and ascending for min player
    )

    if maximizingPlayer:
        maxEval = -math.inf
        for move in legal_moves:
            sim_board, col_idx, row_idx = make_move_minimax(
                board, move, 1
            )  # Max player is always 1
            eval = minimax(
                sim_board, depth - 1, alpha, beta, False, depth_reached=depth_reached
            )
            sim_board = undo_move(sim_board, col_idx, row_idx)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        for move in legal_moves:
            sim_board, col_idx, row_idx = make_move_minimax(
                board, move, -1
            )  # Min player is always -1
            eval = minimax(
                sim_board, depth - 1, alpha, beta, True, depth_reached=depth_reached
            )
            sim_board = undo_move(sim_board, col_idx, row_idx)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def AI(board, player, depth):
    bestMove = None
    bestValue = -math.inf if player == 1 else math.inf
    legal_moves = get_legal_moves(board)
    # order moves by best to worst using the heuristic
    legal_moves.sort(
        key=lambda x: evaluate(make_move(copy.deepcopy(board), x, player)[0], player),
        reverse=(player == 1),  # descending for max player and ascending for min player
    )

    for move in legal_moves:
        # Use a deep copy of the board for simulation
        sim_board = copy.deepcopy(board)
        sim_board, _ = make_move(sim_board, move, player)
        boardValue = minimax(
            sim_board,
            depth - 1,
            -math.inf,
            math.inf,
            (
                AI_PLAYER != 1
            ),  # reversed since next player is always the opposite of the current player
        )

        if player == 1 and (boardValue > bestValue):
            bestValue = boardValue
            bestMove = move

        elif player == -1 and (boardValue < bestValue):
            bestValue = boardValue
            bestMove = move

    if bestMove is None:
        print("AI CANT find a move. likely due to a 0 eval on all moves")
        return get_legal_moves(board)[0]  # failsafe
    return bestMove


def human_vs_ai(who_moves_first=HUMAN_PLAYER, max_depth=4):
    files = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    last_file = files[N - 1]
    board = get_blank_board()
    current_player = who_moves_first
    game_status = 0  # 0 means the game is ongoing

    while game_status == 0 and get_legal_moves(board):
        print_board(board)
        game_status = brute_force_game_over(board)
        if current_player == HUMAN_PLAYER:
            move = None
            while move not in get_legal_moves(board):
                try:
                    print()
                    user_input = input(f"Your move (A-{last_file}): ").strip().upper()
                    move = files.index(user_input)
                except (ValueError, IndexError):
                    print(
                        f"Invalid move. Please choose a column from A to {last_file}."
                    )

            board, game_status = make_move(board, move, HUMAN_PLAYER)
        else:
            print("\nAI is thinking...")
            move = AI(board, AI_PLAYER, max_depth)  # Adjust depth as needed
            board, game_status = make_move(board, move, AI_PLAYER)
            print(f"AI moved: {files[move]}")

        # Switch player
        current_player = AI_PLAYER if current_player == HUMAN_PLAYER else HUMAN_PLAYER
        print(f"EVAL AI: ", evaluate(board, AI_PLAYER))
        print("-----------------------------------")

    print_board(board)

    # Declare the result
    if game_status == HUMAN_PLAYER:
        print("Congratulations! You win!")
    elif game_status == AI_PLAYER:
        print("\nThe AI Won! Better luck next time.")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    human_vs_ai()
