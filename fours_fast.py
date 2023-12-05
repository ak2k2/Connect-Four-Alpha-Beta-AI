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
    """
    Inserts a piece into the specified column.
    Returns the updated board and the row index where the piece was placed.
    """
    col = board[col_num]
    row_idx = None
    for i in range(M - 1, -1, -1):
        if col[i] == 0:
            col[i] = player
            row_idx = i
            break
    board[col_num] = col
    return board, row_idx


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
    board, row_idx = insert_piece_optimized(board, col_num, player)

    # Check if this move resulted in a game over
    status = game_over_optimized(board, col_num, row_idx, player)

    return board, status


def evaluate_board(board):
    # Check for win/loss
    for col in range(N):
        for row in range(M):
            if board[col][row] != 0:
                # Check if the current player has a winning move
                if game_over_optimized(board, col, row, board[col][row]):
                    return 100 if board[col][row] == 1 else -100
    return 0


# Streamlined Example Usage
board = get_blank_board()

# Making moves and checking for game status
moves = [
    (0, 1),
    (1, -1),
    (2, 1),
    (3, -1),
    (3, 1),
    (2, -1),
    (1, 1),
    (1, -1),
    (0, 1),
    (0, 1),
    (0, -1),
]

for col, player in moves:
    board, status = make_move(board, col, player)
    print_board(board)
    if status != 0:
        print(f"Player {player} wins!")
        break

# Print the final board state for visualization
print_board(board)
