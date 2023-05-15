import time

import AI
import Algorithm
from TicTacToe import play

if __name__ == "__main__":

    players = []
    # players.append(AI.QNeural([45, 45]))
    players.append(AI.QNeural())
    # players.append(AI.QTable())
    # players.append(AI.QTable())
    players.append(Algorithm.Random())
    # players.append(Algorithm.Human())
    # players.append(Algorithm.MinMax(1))
    # players.append(Algorithm.MinMax(2))
    # players.append(Algorithm.MinMax(3))
    # players.append(Algorithm.MinMax(4))
    # players.append(Algorithm.MinMax(5))

    t = time.time()
    play(players, 50, 10000, sweep=True, rst=True, league=True)
    print(time.time() - t)


