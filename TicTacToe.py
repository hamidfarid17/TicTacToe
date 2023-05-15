import time
from operator import attrgetter

import numpy as np

from Constant import *


class TicTacToe:

    def __init__(self, player1, player2, WINDOW):
        self.board = np.zeros(9, dtype=int)

        self.turn = 0
        self.winner = None
        self.loser = False
        self.game_over = False
        self.player = [player1, player2]
        if player1.__module__ == 'Human' or player2.__module__ == 'Human':
            self.print = True
        else:
            self.print = PRINT
        self.WINDOW = WINDOW
        if UI:

            pygame.display.set_caption(player1.name + '-' + player2.name)
            pygame.draw.rect(WINDOW, BACKGROUND, pygame.Rect(0, 0, 150, 150))
            for i in range(3):
                for j in range(3):
                    pygame.draw.rect(self.WINDOW, RED, pygame.Rect(i * 50, j * 50, 50, 50), 1)
            fontObj = pygame.font.Font(None, 50)
            self._x_Obj = fontObj.render('X', True, XCOLOUR, None)
            self._o_Obj = fontObj.render('O', True, OCOLOUR, None)
            self._myObj = self._x_Obj
            pygame.display.flip()

    def print_board(self, move):
        if self.print:
            pos = [move % 3 * 50 + 10, move // 3 * 50 + 10]
            self.WINDOW.blit(self._myObj, pos)

    def get_move(self):
        move = self.player[self.turn].get_move(self.board)
        if self.board[move] == 0:
            self.board[move] = 1 if self.turn == 0 else -1
            self.print_board(move)
            self.check_win()
        else:
            self.loser = True
            self.winner = 1 if self.turn == 1 else -1
        self.turn = 1 - self.turn
        if UI:
            pygame.display.flip()
        return move

    def check_win(self):
        for row in range(3):
            if self.board[row * 3 + 0] == self.board[row * 3 + 1] == self.board[row * 3 + 2] and \
                    self.board[row * 3] != 0:
                self.winner = self.board[row * 3]
        for col in range(3):
            if self.board[0 + col] == self.board[3 + col] == self.board[6 + col] and self.board[col] != 0:
                self.winner = self.board[col]
        if self.board[0] == self.board[4] == self.board[8] and self.board[0] != 0:
            self.winner = self.board[0]
        if self.board[2] == self.board[4] == self.board[6] and self.board[2] != 0:
            self.winner = self.board[2]
        jj = [i for i in range(9) if self.board[i] == 0]
        if len(jj) == 0:
            self.game_over = True

    def reward(self):
        if self.loser:
            return 0
        elif self.winner == 1:
            return 1
        elif self.winner == -1:
            return 1
        elif self.game_over:
            return .5
        else:
            return 0

    def play_game(self):
        game_over = False
        r = 0
        while not game_over:
            state = self.board.copy()
            action = self.get_move()
            reward = self.reward()
            self.player[1-self.turn].add_memory(state, action, reward)
            if UI:
                if self._myObj == self._o_Obj:
                    self._myObj = self._x_Obj
                else:
                    self._myObj = self._o_Obj

            if self.winner == 1:
                r = 1
                game_over = True
            elif self.winner == -1:
                game_over = True
                r = -1
            elif self.game_over:
                game_over = True
                r = 0

        return r


def play(player, num, N, sweep=True, rst=True, league=True):
    nn = num
    if UI:
        WINDOW_HEIGHT = 220 + len(player) * 15
        WINDOW_WIDTH = 500

        WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        WINDOW.fill(BACKGROUND)
    else:
        WINDOW = None
    if league:
        MP_list = MP_League_export(len(player), sweep)
    else:
        MP_list = MP_Friendly_export(len(player), sweep)

    while num > 0:
        num -= 1
        for n, (i, j) in enumerate(MP_list):
            play_n_game(player, i, j, n, len(MP_list), N, WINDOW)
        dd = 'Game {:d}'.format(nn - num)
        print('////////////////////////////////////////{:^15}////////////////////////////////////////'.format(dd))
        new_team = sorted(player, key=attrgetter('point', 'win', 'tie'), reverse=True)

        for i in new_team:
            print(
                '/////{:20} match:{:6}  point:{:1.2f}  win:{:6d}  lose:{:6d}  Tie:{:6d}   /////'.format(i.name, i.match,
                                                                                                     i.point / i.match,
                                                                                                     i.win,
                                                                                                     i.lose,
                                                                                                     i.tie))
        print('///////////////////////////////////////////////////////////////////////////////////////////////')
        for i in player:
            i.save()
            if rst:
                i.reset()


def play_n_game(player, p1, p2, num, num_s, N, WINDOW):
    pl = [0, 0, 0]
    tm = time.time()
    if N > 100:
        n = N // 100
    else:
        n = 1
    for i in range(N):
        tt = TicTacToe(player[p1], player[p2], WINDOW)

        m = tt.play_game()
        tie = tt.game_over and (tt.winner is None)
        player[p1].replay(tie)
        player[p2].replay(tie)

        if m == 1:
            player[p1].win += 1
            player[p1].point += 1
            player[p2].lose += 1
            pl[0] += 1
        elif m == -1:
            player[p1].lose += 1
            player[p2].win += 1
            player[p2].point += 1
            pl[1] += 1
        else:
            player[p1].tie += 1
            player[p2].tie += 1
            player[p2].point += .5
            player[p1].point += .5

            pl[2] += 1
        player[p1].match += 1
        player[p2].match += 1
        if i % n == n - 1:
            t2 = time.time()
            print('\r{:2}/{:2}-p{:3.0%} time:{:13}'.
                  format(num + 1, num_s, (i + 1) / N, change_time(t2 - tm)), end='  ***  ')
            print('{:10}:{:4.2%}, {:10}:{:4.2%}, Tie:{:4.2%}'.
                  format(player[p1].name, (pl[0]) / (i + 1), player[p2].name, (pl[1]) / (i + 1),
                         (pl[2]) / (i + 1)), end='')
        st = [[player[p1].name, '{:.2%}'.format(pl[0] / (i + 1))],
              [player[p2].name, '{:.2%}'.format(pl[1] / (i + 1))],
              ['Tie', '{:.2%}'.format(pl[2] / (i + 1))]]
        if UI:

            font_obj = pygame.font.Font(None, 20)
            pygame.draw.rect(WINDOW, BACKGROUND, pygame.Rect(200, 0, 200, 100))
            _Obj = font_obj.render('-----------Current {:2.0%}-----------'.format((i + 1) / N), True, XCOLOUR, None)
            WINDOW.blit(_Obj, (200, 0))
            for ii, (s, t) in enumerate(st):
                _Obj = font_obj.render(s + ' :', True, OCOLOUR, None)
                WINDOW.blit(_Obj, (200, 20 + 15 * ii))
                _Obj = font_obj.render(t, True, OCOLOUR, None)
                WINDOW.blit(_Obj, (300, 20 + 15 * ii))
            pygame.display.flip()

    print('\r{:2}/{:2}-full time:{:13}  ***  {:20}:{:4.2%}, {:20}:{:4.2%}, Tie:{:4.2%}'.
          format(num + 1, num_s, change_time(time.time() - tm),
                 player[p1].name, pl[0] / N, player[p2].name, pl[1] / N, pl[2] / N))

    st = [[player[p1].name, '{:.2%}'.format(pl[0] / N)],
          [player[p2].name, '{:.2%}'.format(pl[1] / N)],
          ['Tie', '{:.2%}'.format(pl[2] / N)]]

    if UI:

        font_obj = pygame.font.Font(None, 20)
        pygame.draw.rect(WINDOW, BACKGROUND, pygame.Rect(200, 100, 200, 200))
        _Obj = font_obj.render('----------------Last----------------', True, XCOLOUR, None)
        WINDOW.blit(_Obj, (200, 100))
        for ii, (s, t) in enumerate(st):
            _Obj = font_obj.render(s + ' :', True, OCOLOUR, None)
            WINDOW.blit(_Obj, (200, 120 + 15 * ii))
            _Obj = font_obj.render(t, True, OCOLOUR, None)
            WINDOW.blit(_Obj, (300, 120 + 15 * ii))

        new_team = sorted(player, key=attrgetter('point', 'win', 'tie'), reverse=True)
        st = [['Name', 'Match', 'Point', 'Win', 'Lose', 'Tie']]
        for i in new_team:
            st.append([i.name, str(i.match), '{:.3}'.format(i.point / i.match if i.match > 0. else 0.),
                       str(i.win), str(i.lose), str(i.tie)])

        font_obj = pygame.font.Font(None, 20)
        pygame.draw.rect(WINDOW, BACKGROUND, pygame.Rect(0, 180, WINDOW.get_width(), WINDOW.get_height() - 180))
        for ii, s in enumerate(st):
            color = XCOLOUR if ii == 0 else OCOLOUR
            for jj, t in enumerate(s):
                _Obj = font_obj.render(t, True, color, None)
                WINDOW.blit(_Obj, ((10 if jj < 2 else 50) + (120 if jj == 1 else 80) * jj, 200 + 15 * ii))

        pygame.display.flip()


