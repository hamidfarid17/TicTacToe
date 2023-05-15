import configparser
import os
import random

import pygame
import torch


def str2bool(v):
    return v.lower() == 'true'


# Colours
BACKGROUND = (255, 255, 255)
XCOLOUR = (200, 100, 0)
OCOLOUR = (100, 200, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

device = torch.device("cpu")

DEFAULT_PRINT = 'False'
DEFAULT_UI = 'False'
DEFAULT_QL_GAMMA = 0.7
DEFAULT_AI_GAMMA = 1.0
DEFAULT_LR = 1e-3
DEFAULT_AI_LR = 1e-3

path = './Checkpoint'
# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)
# Create a new configparser instance and load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')
# Get the values of the configuration variables, using default values if not available
PRINT = str2bool(config.get('DEFAULT', 'PRINT', fallback=DEFAULT_PRINT))
UI = str2bool(config.get('DEFAULT', 'UI', fallback=DEFAULT_UI))
QL_GAMMA = float(config.get('DEFAULT', 'QL_GAMMA', fallback=DEFAULT_QL_GAMMA))
AI_GAMMA = float(config.get('DEFAULT', 'AI_GAMMA', fallback=DEFAULT_AI_GAMMA))
LR = float(config.get('DEFAULT', 'LR', fallback=DEFAULT_LR))

if len(config.defaults()) == 0:
    # Update config variable with given data from user
    config['DEFAULT'] = {
        'PRINT': str(PRINT),
        'QL_GAMMA': str(QL_GAMMA),
        'AI_GAMMA': str(AI_GAMMA),
        'LR': str(LR)
    }

    # Saving the configuration info to config file for further use
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

if UI:
    pygame.init()
    # Game Setup
    FPS = 60
    fpsClock = pygame.time.Clock()


def change_time(t):
    s = ''
    tt = t
    if tt > 3600:
        m = t // 3600
        s = '{:d}h:'.format(int(m))
        tt -= m * 3600
    if tt > 60:
        m = tt // 60
        s += '{:d}m:'.format(int(m))
        tt -= m * 60
    s += '{:.2f}s'.format(tt)
    return s


def MP_League_export(number, sweep=True):
    l = list(range(0, number))
    week = number - 1 if number % 2 == 0 else number
    MP = []
    i = 0
    while i < week:
        m = l.copy()
        cash = []
        loop_test = 0
        while len(m) > 1:
            a, b = random.choices(m, k=2)

            if a != b:
                if (a, b) not in MP and (b, a) not in MP:
                    cash.append((a, b))
                    m.remove(a)
                    m.remove(b)
                elif len(m) < 4:
                    loop_test += 1
                    if loop_test == 20 or len(l) < 4:
                        MP.clear()
                        cash.clear()
                        i = -1

                        break
                    else:
                        a, b = cash.pop()
                        m.append(a)
                        m.append(b)

        MP += cash
        i += 1
    if sweep:
        cash = []
        for a, b in MP:
            cash.append((b, a))
            # cash.append((a, b))
        MP += cash
    return MP


def MP_Friendly_export(number, sweep=True):
    l = list(range(1, number))
    mp = []
    for i in range(number - 1):
        mp.append((0, l[i]))

    if sweep:
        cash = []
        for a, b in mp:
            cash.append((b, a))
        mp += cash
    return mp
