import pytest
import sys
import pathlib

# Add the parent directory to the path so we can import connect_four
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from connect_four import *

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
    (0, 1),
]

for col, player in moves:
    board, status = make_move(board, col, player)
    print_board(board)
    print(get_legal_moves(board))
    print(f"eval: {evaluate(board, AI_PLAYER)}")
    if status != 0:
        print(f"Player {player} wins!")
        break

# Print the final board state for visualization
print_board(board)
