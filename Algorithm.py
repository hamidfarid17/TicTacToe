import sys

import numpy as np
import pygame
from pygame.locals import *


class Human:
    def __init__(self):
        self.name = 'human   '
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def get_move(self, board):
        move = -1
        while move == -1:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == BUTTON_LEFT:
                        if event.pos[0] < 150 and event.pos[1] < 150:
                            move = event.pos[0] // 50 + event.pos[1] // 50 * 3
                            if board[move] != 0:
                                move = -1

        return move

    def replay(self, tie):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def reset(self):
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def add_memory(self, state, action, reward):
        pass


class MinMax:
    def __init__(self, skill=None):
        if skill is None:
            self.skill = 5
        else:
            self.skill = skill
        self.name = 'MinMax_' + str(self.skill)
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def get_move(self, board):
        turn = sum(board)
        action_list = [i for i in range(9) if board[i] == 0]
        _, mv = minmax(board, turn, turn, action_list,
                       0 if len(action_list) == 9 else self.skill)
        return np.random.choice(mv)

    def replay(self, tie):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def reset(self):
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def add_memory(self, state, action, reward):
        pass


def check_rew(board, r):
    for row in range(3):
        if board[row * 3 + 0] == board[row * 3 + 1] == board[row * 3 + 2] and board[row * 3] != 0:
            return r
    for col in range(3):
        if board[0 + col] == board[3 + col] == board[6 + col] and board[col] != 0:
            return r
    if board[0] == board[4] == board[8] and board[0] != 0:
        return r
    if board[2] == board[4] == board[6] and board[2] != 0:
        return r
    jj = [i for i in range(9) if board[i] == 0]
    if len(jj) == 0:
        return .5
    return 0


def minmax(board, my_turn, new_turn, action_list, deep):
    mm = []
    nt = []
    r = 1 if my_turn == new_turn else -1
    for i in action_list:
        b = board.copy()
        b[i] = 1 if new_turn == 0 else -1
        cr = check_rew(b, r)
        if cr == 0 and deep != 0:
            ac_list = [j for j in range(9) if b[j] == 0]
            cr, _ = minmax(b, my_turn, 1 - new_turn, ac_list, deep - 1)
        mm.append(cr)
        nt.append(i)

    mx = max(mm) if my_turn == new_turn else min(mm)
    return mx, [nt[j] for j in range(len(mm)) if mm[j] == mx]


class Random:
    def __init__(self):
        self.name = 'random  '
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def get_move(self, board):
        action_list = [i for i in range(9) if board[i] == 0]
        return np.random.choice(action_list)

    def replay(self, tie):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def reset(self):
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def add_memory(self, state, action, reward):
        pass
