M = 6
N = 7
IN_A_ROW = 4

# -1 is RED
# 0 is BLANK
# 1 is YELLOW


def print_board(board):
    """
    Prints a Connect Four board with column labels (A-G) and neat boxes.
    -1 represents a red piece, 0 an empty square, and 1 a yellow piece.
    """
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


def get_blank_board():
    return [[0 for i in range(M)] for j in range(N)]


def get_rows(board: list) -> list:
    rows = []
    for j in range(len(board[0])):
        row = []
        for i in range(len(board)):
            row.append(board[i][j])
        rows.append(row)
    return rows


def insert_piece(board: list, col_num: int, player: int):
    col = board[col_num]
    for i in range(M - 1, -1, -1):
        if col[i] == 0:
            col[i] = player
            break
    board[col_num] = col
    return board


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


def get_valid_diagonals(board: list):
    max_col = M
    max_row = N
    cols = [[] for _ in range(max_col)]
    rows = [[] for _ in range(max_row)]
    fdiag = [[] for _ in range(max_row + max_col - 1)]
    bdiag = [[] for _ in range(len(fdiag))]
    min_bdiag = -max_row + 1

    for x in range(max_col):
        for y in range(max_row):
            cols[x].append(board[y][x])
            rows[y].append(board[y][x])
            fdiag[x + y].append(board[y][x])
            bdiag[x - y - min_bdiag].append(board[y][x])

    gfd = [d for d in fdiag if len(d) >= 4] + [d for d in bdiag if len(d) >= 4]

    return gfd


def game_over(board):
    gfd = get_valid_diagonals(board)
    rows = get_rows(board)

    # Check each row, column, and diagonal
    for collection in [rows, board, gfd]:
        for line in collection:
            status = in_a_row_met(line)
            if status != 0:
                print("WINNER")
                return status

    print("No winner")
    return 0


board = get_blank_board()
board = insert_piece(board, 0, 1)
board = insert_piece(board, 1, 1)
board = insert_piece(board, 2, 1)
game_over(board)

board = insert_piece(board, 3, 1)
game_over(board)
print_board(board)
