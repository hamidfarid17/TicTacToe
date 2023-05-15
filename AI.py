import pickle
import random
from collections import deque

import numpy as np
import torch
from torch import nn

from Constant import AI_GAMMA, LR, device


class QTable:
    def __init__(self, file_name=None, ):
        self.policy = {}
        self.my_lr = .5
        self.max = 1.0
        self.epsilon = 1.0
        if file_name is None:
            file_name = 'policy'
        self.name = 'QL_' + str(file_name)
        self.load()
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0
        self.buffer = deque()

    def save(self):
        mx = self.lose / self.match
        if self.max != 0 or mx != 0:
            self.max = mx
            with open('Checkpoint/' + self.name + '.ql', 'wb') as f:
                pickle.dump([self.policy, self.my_lr, self.max, self.epsilon], f)
            print('++++++++++++++++  Save {}.ql  = {}  , lose={:.2%}  , epsilon={}'.
                  format(self.name, len(self.policy), self.max, self.epsilon))

    def load(self):
        try:
            with open('Checkpoint/' + self.name + '.ql', 'rb') as f:
                self.policy, self.my_lr, self.max, self.epsilon = pickle.load(f)
            print('++++++++++++++++  Load {}.ql  = {}  , lose={:.2%}  , epsilon={}'.
                  format(self.name, len(self.policy), self.max, self.epsilon))
        except:
            print('!!!!!!!!!!!!!!!! {}.ql not load'.format(self.name))

    def get_move(self, board):
        action_list = [i for i in range(9) if board[i] == 0]
        if random.random() < self.epsilon:
            action = int(random.choice(action_list))
        else:
            mv = []
            for jj in action_list:
                b = board.copy()
                _, r = self.get_value(b, jj)
                mv.append(r)
            action = action_list[np.random.choice(np.where(mv == np.amax(mv))[0])]
        return action

    def replay(self, tie):
        if self.max > 0:
            new_state, action, reward = self.buffer.pop()
            if tie:
                reward = .2
            new_sts, new_st_value = self.get_value(new_state, action)
            self.policy[new_sts] = float(reward)  # (1 - self.my_lr) * new_st_value + self.my_lr * float(reward)

            while len(self.buffer) > 0:
                state, action, reward = self.buffer.pop()
                action_list = [i for i in range(9) if new_state[i] == 0]
                mv = []
                for jj in action_list:
                    b = new_state.copy()
                    _, r = self.get_value(b, jj)
                    mv.append(r)

                rew = np.amax(mv)

                new_sts, st_value = self.get_value(state, action)

                self.policy[new_sts] = st_value * (1 - self.my_lr) + self.my_lr * rew
                new_state = state.copy()

            if self.epsilon > 0.0:
                self.epsilon -= 1e-5
            else:
                self.epsilon = 0.0

            if self.my_lr <= .1:
                self.my_lr = 1e-5
            else:
                self.my_lr *= 1 - 1e-6
        else:
            self.buffer.clear()

    def reset(self):
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def add_memory(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def get_value(self, board, action):
        turn = sum(board)
        new_board = board.copy()
        new_board[action] = 1 if turn == 0 else -1
        for i in range(8):
            s, r = self.get_value_(new_board, i)
            if r is not None:
                return s, r
        return s, 0

    def get_value_(self, board, ii):
        board = np.reshape(board, (-1, 3))
        if ii == 0:
            board = np.rot90(board)
        elif ii == 1:
            board = np.rot90(board, 2)
        elif ii == 2:
            board = np.rot90(board, 3)
        elif ii == 3:
            board = np.flipud(board)
        elif ii == 4:
            board = np.fliplr(board)
        elif ii == 5:
            board = np.flipud(np.rot90(board))
        elif ii == 6:
            board = np.fliplr(np.rot90(board))
        board = board.flatten()
        sts = board.tobytes()
        return sts, self.policy[sts] if sts in self.policy.keys() else None


class QNeural:
    def __init__(self, layers=None):
        self.my_lr = LR
        self.max = 1.0
        self.epsilon = 1.0
        if layers is None:
            layers = [36, 36]
        name = ''
        for i in layers:
            name += str(i) + '_'
        self.name = 'AI_' + name[:-1]
        self.net = DQN(layers)
        self.load()
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.my_lr)
        self.loss_function = nn.MSELoss()
        self.buffer = deque()

    def get_move(self, board):

        if random.random() < self.epsilon:
            action_list = [i for i in range(9) if board[i] == 0]
            action = int(random.choice(action_list))
        else:
            state = torch.tensor(np.array([board]), dtype=torch.float).to(device)
            action_probs = self.net.forward(state)
            _, action = torch.max(action_probs, dim=1)
            action = int(action.item())
        return action

    def replay(self, tie):
        if self.max > 0:
            state, action, reward = self.buffer.pop()
            if tie:
                reward = .8
            state_v = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            output = self.net.forward(state_v)
            target = output.clone().detach()
            illegal_list = [i for i in range(9) if state[i] != 0]
            for i in illegal_list:
                target[0][i] = 0.0
            target[0][action] = float(reward)
            new_state = state
            loss = self.loss_function(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(reward)
            while len(self.buffer) > 0:
                state, action, reward = self.buffer.pop()
                state_v = torch.tensor(np.array([new_state]), dtype=torch.float).to(device)
                next_q_values = self.net.forward(state_v)
                rew = torch.max(next_q_values).item()

                state_v = torch.tensor(np.array([state]), dtype=torch.float).to(device)
                output = self.net.forward(state_v)
                target = output.clone().detach()
                illegal_list = [i for i in range(9) if state[i] != 0]
                for i in illegal_list:
                    target[0][i] = 0.
                target[0][action] = rew * AI_GAMMA
                new_state = state

                loss = self.loss_function(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.epsilon > 0.0:
                self.epsilon -= 5e-7
            else:
                self.epsilon = 0.0
        else:
            self.buffer.clear()

    def save(self, pr=True):
        mx = self.lose / self.match
        if self.max != 0 or mx != 0:
            self.max = mx
            torch.save({
                'model_state_dict': self.net.state_dict(),
                'lr': self.my_lr,
                'epsilon': self.epsilon,
                'max': self.max}, 'Checkpoint/' + self.name + '.ai')
            print('++++++++++++++++  Save {}.ai  =  {} , lose={:.2%} , epsilon ={}'.
                  format(self.name, self.my_lr, self.max, self.epsilon))

    def load(self):
        try:
            checkpoint = torch.load('Checkpoint/' + self.name + '.ai')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.my_lr = checkpoint['lr']
            self.epsilon = checkpoint['epsilon']
            self.max = checkpoint['max']
            print('load {}.ai  =  {} , lose={:.2%} , epsilon ={}'.
                  format(self.name, self.my_lr, self.max, self.epsilon))
        except:
            print('{}.ai not load'.format(self.name))

    def reset(self):
        self.win = 0
        self.lose = 0
        self.point = 0
        self.tie = 0
        self.match = 0

    def add_memory(self, state, action, reward):
        self.buffer.append((state, action, reward))


class DQN(nn.Module):
    def __init__(self, layer):
        super(DQN, self).__init__()
        n_in = 9
        n_out = 9
        layers = []
        for l in layer:
            layers.append(nn.Linear(n_in, l))
            layers.append(nn.ReLU())
            n_in = l
        layers.append(nn.Linear(n_in, n_out))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

