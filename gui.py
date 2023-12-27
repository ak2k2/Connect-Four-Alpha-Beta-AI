import math
import sys
import time

import pygame

from constants import AI_PLAYER, HUMAN_PLAYER, IN_A_ROW, M, N
from game import *  # Importing your Connect Four logic

# Pygame Initialization
pygame.init()


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


# Constants for Pygame
COLUMN_COUNT = N
ROW_COUNT = M
SQUARESIZE = 100
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = ROW_COUNT * SQUARESIZE  # Extra space for the top row

# Colors
BLUE = (100, 0, 185)
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
                (c * SQUARESIZE, HEIGHT - (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE),
            )
            pygame.draw.circle(
                screen,
                BLACK,
                (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
                ),
                SQUARESIZE // 2 - 5,
            )

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[c][ROW_COUNT - r - 1] == AI_PLAYER:
                pygame.draw.circle(
                    screen,
                    YELLOW,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    SQUARESIZE // 2 - 5,
                )
            elif board[c][ROW_COUNT - r - 1] == HUMAN_PLAYER:
                pygame.draw.circle(
                    screen,
                    RED,
                    (
                        int(c * SQUARESIZE + SQUARESIZE / 2),
                        HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
                    ),
                    SQUARESIZE // 2 - 5,
                )
    pygame.display.update()


def human_vs_ai():
    board = get_blank_board()
    draw_board(board)
    pygame.display.update()

    game_over = False
    turn = HUMAN_PLAYER

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if turn == HUMAN_PLAYER and not game_over:
            pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN)  # Enable mouse clicks

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    xpos = event.pos[0]
                    col = int(math.floor(xpos / SQUARESIZE))
                    legal_moves = get_legal_moves(board)

                    if col in legal_moves:
                        board, status = make_move(board, col, HUMAN_PLAYER)
                        draw_board(board)
                        if status != 0:
                            game_over = True
                        else:
                            turn = AI_PLAYER
                            pygame.time.wait(100)  # Wait for half a second
                        break  # Break after the move is made

        elif turn == AI_PLAYER and not game_over:
            pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)  # Disable mouse clicks

            # AI makes its move
            col = AI(board, AI_PLAYER, 4)  # Adjust the depth as needed
            board, status = make_move(board, col, AI_PLAYER)
            draw_board(board)
            if status != 0:
                game_over = True
            else:
                turn = HUMAN_PLAYER

    # Post-game messages and cleanup
    if status != 0:
        print(f"Game Over! {'Yellow' if status == 1 else 'Red'} wins!")
    else:
        print("Draw! No More Moves")

    pygame.time.wait(1000)
    pygame.quit()


def ai_vs_ai():
    board = get_blank_board()
    draw_board(board)
    pygame.display.update()

    game_over = False
    turn = HUMAN_PLAYER

    while not game_over:  # game in progress
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # exit game gracefully
                sys.exit()

            if turn == HUMAN_PLAYER:
                col = AI(board, HUMAN_PLAYER, 3)  # Adjust the depth as needed
                time.sleep(1)
                board, status = make_move(board, col, HUMAN_PLAYER)
                pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN)
                if status != 0:
                    game_over = True
            turn = AI_PLAYER

        if turn == AI_PLAYER and not game_over and get_legal_moves(board):
            # disable the mouse from clicking
            pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
            col = AI(board, AI_PLAYER, 3)  # Adjust the depth as needed
            board, status = make_move(board, col, AI_PLAYER)
            pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN)
            if status != 0:
                game_over = True
            turn = HUMAN_PLAYER

            draw_board(board)

    print(f"Game Over! Player {status} wins!")

    # Game Over
    pygame.time.wait(1000)
    pygame.quit()


if __name__ == "__main__":
    human_vs_ai()
    # ai_vs_ai()
