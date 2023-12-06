import pygame
import sys
import math
from connect_four import *  # Importing your Connect Four logic

# Pygame Initialization
pygame.init()

# Constants for Pygame
COLUMN_COUNT = N
ROW_COUNT = M
SQUARESIZE = 100
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE  # Extra space for the top row

# Colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# Setting up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect Four AI")


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(
                screen,
                BLUE,
                (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE),
            )
            pygame.draw.circle(
                screen,
                BLACK,
                (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2),
                ),
                SQUARESIZE // 2 - 5,
            )

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[c][r] == AI_PLAYER:
                pygame.draw.circle(
                    screen,
                    YELLOW,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        int((ROW_COUNT - r) * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    SQUARESIZE // 2 - 5,
                )
            elif board[c][r] == HUMAN_PLAYER:
                pygame.draw.circle(
                    screen,
                    RED,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        int((ROW_COUNT - r) * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    SQUARESIZE // 2 - 5,
                )
    pygame.display.update()


board = get_blank_board()
draw_board(board)
pygame.display.update()

game_over = False
turn = HUMAN_PLAYER

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            xpos = event.pos[0]
            col = int(math.floor(xpos / SQUARESIZE))

            if turn == HUMAN_PLAYER and get_legal_moves(board):
                board, status = make_move(board, col, HUMAN_PLAYER)
                if status != 0:
                    game_over = True
                turn = AI_PLAYER

            draw_board(board)

        if turn == AI_PLAYER and not game_over and get_legal_moves(board):
            col = AI(board, AI_PLAYER, 4)  # Adjust the depth as needed
            board, status = make_move(board, col, AI_PLAYER)
            if status != 0:
                game_over = True
            turn = HUMAN_PLAYER

            draw_board(board)

# Game Over
pygame.time.wait(3000)
pygame.quit()
