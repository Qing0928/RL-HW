#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

# 棋盤
class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        '''
            初始矩陣
            000,
            000,
            000
        '''
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))  # 建立一個全為0的矩陣當作棋盤
        self.winner = None      # 現在這局的勝者
        self.hash_val = None    # 棋局對應的 hash 值，方便用來尋找棋局
        self.end = None         # 現在這局是否結束?

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            # np.nditer()會返回一個擁有矩陣中所有的值迭代器
            for i in np.nditer(self.data): 
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    # 計算棋局是否結束，這局的結果會暫存起來，利用hash值來尋找
    # !!可能會有平局!!
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = 'O'
                elif self.data[i, j] == -1:
                    token = 'X'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


'''
get_all_states_impl()
用來取得所有可能的棋局，並計算棋局的狀態
會存在 all_states (dict型態)
透過hash值就可以找到每個棋局的資訊
'''
def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


def get_all_states():
    # 1 為先攻
    current_symbol = 1  
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

# all possible board configurations
all_states = get_all_states()

# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    # 公式 Q(st, at) ← (1 - ɑ)Q(st-1, at-1) + ɑ[rt + ɼmaxQ(st+1, a)]
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()   # 儲存各個棋局的估算勝率，即 value
        self.step_size = step_size  # 步長參數，用於控制回朔更新的步長
        self.epsilon = epsilon      # 探索的概率
        self.states = []            # 紀錄本局經過的棋局
        self.greedy = []
        self.symbol = 0             # 先手為 1, 後手為 -1
        
    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        # 初始化預測表
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        states = [state.hash() for state in self.states]

        # 按逆序對本局所有按 greedy 原則落子局面的預測值進行更新
        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1] # 上一個棋局
        next_states = []        # 記錄所有可能的落子位置對應棋局的 hash 值
        next_positions = []     # 記錄所有可能的落子位置
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon: # 進行試探落子
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False # 更新標誌位
            return action

        # 按 greedy 原則落子，選擇預測勝率最高的地方落子
        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]   # 落子位置
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# 裁判，用來訓練
class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1:Player, player2:Player):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        '''
        有兩個玩家，如果是電腦先攻，會初始化預測表
        '''
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        # not use in this example
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    # p1跟p2輪流動作
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()   # 返回一個生成器
        self.reset()                    # 初始化電腦玩家的狀態
        current_state = State()         # 初始化棋局
        
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        # if print_state:
        #     current_state.print_state()
        while True:
            player = next(alternator)   # 獲取下一個玩家
            i, j, symbol = player.act() # 玩家落子
            next_state_hash = current_state.next_state(i, j, symbol).hash() # 獲取落子後棋盤的哈希值
            current_state, is_end = all_states[next_state_hash] # 由hash獲取棋盤信息
            self.p1.set_state(current_state)    # 記錄當前棋局
            self.p2.set_state(current_state)
            # if print_state:
            #     current_state.print_state()
            if is_end:
                if print_state:
                    current_state.print_state()
                return current_state.winner


# human interface
# 棋盤位置對照表，遊戲時輸入數字即可
# | 7 | 8 | 9 |
# | 4 | 5 | 6 |
# | 1 | 2 | 3 |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None  # 標記誰先攻
        self.keys = ['7', '8', '9', '4', '5', '6', '1', '2', '3']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol


def train_p1_first(epochs, print_every_n=1000):
    # 訓練時採用左右互搏 (self-play)
    player1 = Player(epsilon=0.01, step_size=0.1)
    player2 = Player(epsilon=0.8, step_size=0.1)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0

    player1_win = 0.0
    player2_win = 0.0
    tie_numbers = 0

    p1_win = []
    p2_win = []
    tie_win = []
    total = []
    print("-----訓練模式-----")
    print("Player1 固定先攻")
    print("-----Player1 訓練參數-----")
    print(f"攻擊順序:{player1.symbol}")
    print(f"探索機率:{player1.epsilon}")
    print(f"學習率:{player1.step_size}")
    print("-----Player2 訓練參數-----")
    print(f"攻擊順序:{player2.symbol}")
    print(f"探索機率:{player2.epsilon}")
    print(f"學習率:{player2.step_size}")
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            tie_numbers += 1
        if i % 10 == 0:
            p1_win += [player1_win]
            p2_win += [player2_win]
            tie_win += [tie_numbers]
            total += [i]
        if i % print_every_n == 0:
            print(f"Epoch {i}, player1 獲勝數:{player1_win}, player2 獲勝數:{player2_win}, 平局數:{tie_numbers}")
        player1.backup()
        player2.backup()

        judger.reset()
    
    print("-----------------------------------------")
    player1_win_rate = round((player1_win / epochs)*100 , 2)
    player2_win_rate = round((player2_win / epochs)*100 , 2)
    tie_rate = round((tie_numbers / epochs)*100, 2)
    print(f"總次數:{epochs}\nplayer1 勝率:{player1_win_rate}%\nplayer2 勝率:{player2_win_rate}%\n平局:{tie_rate}%")
    player1.save_policy()
    player2.save_policy()

    plt.plot(total, p1_win, 'r', label="Player1 Wins")
    plt.plot(total, p2_win, 'g', label="Player2 Wins")
    plt.plot(total, tie_win, 'b', label="Tie")
    plt.xlabel("Round")
    plt.ylabel("Win")
    plt.legend()
    plt.show()

def train_half(epochs, print_every_n=1000):
    # 訓練時採用左右互搏 (self-play)
    player1 = Player(epsilon=0.01)  
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0

    player1_win = 0.0
    player2_win = 0.0
    tie_numbers = 0

    p1_win = []
    p2_win = []
    tie_win = []
    total = []

    print("-----訓練模式-----")
    print("Player2 訓練次數減半")
    print("-----Player1 訓練參數-----")
    print(f"攻擊順序:{player1.symbol}")
    print(f"探索機率:{player1.epsilon}")
    print(f"學習率:{player1.step_size}")
    print("-----Player2 訓練參數-----")
    print(f"攻擊順序:{player2.symbol}")
    print(f"探索機率:{player2.epsilon}")
    print(f"學習率:{player2.step_size}")

    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            tie_numbers += 1
        if i % 10 == 0:
            p1_win += [player1_win]
            p2_win += [player2_win]
            tie_win += [tie_numbers]
            total += [i]
        if i % print_every_n == 0:
            print(f"Epoch {i}, player1 獲勝數:{player1_win}, player2 獲勝數:{player2_win}, 平局數:{tie_numbers}")
        player1.backup()
        if i <= epochs / 2:
            player2.backup()
        judger.reset()
    
    print("-----------------------------------------")
    print(f"總次數:{epochs}\nplayer1 勝率:{round(player1_win / epochs, 2)*100}%\nplayer2 勝率:{round(player2_win / epochs, 2)*100}%\n平局:{round(tie_numbers / epochs, 2)*100}%")
    player1.save_policy()
    player2.save_policy()

    plt.plot(total, p1_win, 'r', label="Player1 Wins")
    plt.plot(total, p2_win, 'g', label="Player2 Wins")
    plt.plot(total, tie_win, 'b', label="Tie")
    plt.xlabel("Round")
    plt.ylabel("Win")
    plt.legend()
    plt.show()

def train_take_turn(epochs, print_every_n=1000):
    # 訓練時採用左右互搏 (self-play)
    player1 = Player(epsilon=0.01, step_size=0.1)
    player2 = Player(epsilon=0.01, step_size=0.1)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0

    player1_win = 0.0
    player2_win = 0.0
    tie_numbers = 0

    p1_win = []
    p2_win = []
    tie_win = []
    total = []
    print("-----訓練模式-----")
    print("Player1、2 輪流先攻")
    print("-----Player1 訓練參數-----")
    print(f"攻擊順序:{player1.symbol}")
    print(f"探索機率:{player1.epsilon}")
    print(f"學習率:{player1.step_size}")
    print("-----Player2 訓練參數-----")
    print(f"攻擊順序:{player2.symbol}")
    print(f"探索機率:{player2.epsilon}")
    print(f"學習率:{player2.step_size}")
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            tie_numbers += 1
        if i % 10 == 0:
            p1_win += [player1_win]
            p2_win += [player2_win]
            tie_win += [tie_numbers]
            total += [i]
        if i % print_every_n == 0:
            print(f"Epoch {i}, player1 獲勝數:{player1_win}, player2 獲勝數:{player2_win}, 平局數:{tie_numbers}")
        player1.backup()
        player2.backup()
        player1.set_symbol(player1.symbol *-1)
        player1.set_symbol(player2.symbol *-1)

        judger.reset()
    
    print("-----------------------------------------")
    player1_win_rate = round((player1_win / epochs)*100 , 2)
    player2_win_rate = round((player2_win / epochs)*100 , 2)
    tie_rate = round((tie_numbers / epochs)*100, 2)
    print(f"總次數:{epochs}\nplayer1 勝率:{player1_win_rate}%\nplayer2 勝率:{player2_win_rate}%\n平局:{tie_rate}%")
    player1.save_policy()
    player2.save_policy()

    plt.plot(total, p1_win, 'r', label="Player1 Wins")
    plt.plot(total, p2_win, 'g', label="Player2 Wins")
    plt.plot(total, tie_win, 'b', label="Tie")
    plt.xlabel("Round")
    plt.ylabel("Win")
    plt.legend()
    plt.show()

# 測試 AI
def compete(turns:int):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()

    player1_win = 0.0
    player2_win = 0.0
    tie_numbers = 0

    p1_win = []
    p2_win = []
    tie_win = []
    total = []

    for i in range(1, turns + 1):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        elif winner == -1:
            player2_win += 1
        else:
            tie_numbers += 1
        if i % 25 == 0:
            p1_win += [player1_win]
            p2_win += [player2_win]
            tie_win += [tie_numbers]
            total += [i]
        #player1.set_symbol(player1.symbol *-1)
        #player1.set_symbol(player2.symbol *-1)
        judger.reset()

    print(f"總對局:{turns}, player1 勝利:{player1_win}, player2 勝利:{player2_win}, 平局:{tie_numbers}")

    plt.plot(total, p1_win, 'r', label="Player1 Wins")
    plt.plot(total, p2_win, 'g', label="Player2 Wins")
    plt.plot(total, tie_win, 'b', label="Ties")
    plt.xlabel("Round")
    plt.ylabel("Win")
    plt.title("AI Duel")
    plt.legend()
    plt.show()


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    # while True:
    player1 = HumanPlayer()
    player2 = Player(epsilon=0) # 真實對戰時，不進行試探
    judger = Judger(player1, player2)

    player2.load_policy()
    winner = judger.play(print_state = True)
    if winner == player2.symbol:
        print("You lose!")
    elif winner == player1.symbol:
        print("You win!")
    else:
        print("It is a tie!")


if __name__ == '__main__':
    '''
    try:
        Path("policy_first.bin").unlink()
        Path("policy_second.bin").unlink()
    except Exception as e:
        print(e.with_traceback)
    '''
    
    train_turns = 5000
    while True:
        key = input('Train(t), Compete(c), Play(p) or Quit(q): ')
        if key == 't':
            mode = input('一般訓練(1)\n訓練次數減半(2)\n輪流先攻(3)\nTrain Mode:')
            if mode == '1':
                train_p1_first(train_turns)
            elif mode == '2':
                train_half(train_turns)
            elif mode == '3':
                train_take_turn(train_turns)
        elif key == 'c':
            compete(100)
        elif key == 'p':
            play()
        elif key == 'q':
            break