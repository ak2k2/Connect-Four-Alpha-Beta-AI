from constants import *


def get_blank_board():
    return [[0 for _ in range(M)] for _ in range(N)]


def in_a_row_met(arr: list, in_a_row=4):
    """
    Check if there are 'in_a_row' consecutive pieces from the same player.
    Return -1 for red wins, 1 for yellow wins, or 0 for no win.
    """
    streak = 0
    last_player = 0

    for p in arr:
        if p == last_player:
            streak += 1
            if streak == in_a_row:
                return p  # Current player wins
        else:
            streak = 1 if p != 0 else 0
            last_player = p

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
