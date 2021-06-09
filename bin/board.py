'''
棋盘:
- 控制走子和游戏规则
- 将当前局面特征向量化
- 判断输赢
棋盘规则：
- 3*3
- 每人3子
- 落子规则
	(x1, y1)->(x2, y2) 走子
	(x1, y1)	落子
- 胜负规则
	三子相连
'''

import numpy as np


class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 3))
        self.height = int(kwargs.get('height', 3))
        self.states = {}  # 存放每个落子点是谁的子，key为棋盘列表形式的每个棋子对应的索引
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 3))
        self.players = [1, 2]  # player1 and player2

        self._down_actions = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
        self._move_actions = ['0001', '0010', '0011',
                              '0100', '0102', '0111',
                              '0201', '0212', '0211',
                              '1000', '1020', '1011',
                              '1100', '1101', '1102', '1110', '1112', '1120', '1121', '1122',
                              '1202', '1211', '1222',
                              '2010', '2011', '2021',
                              '2120', '2111', '2122',
                              '2221', '2211', '2212']
        self.actions = self._down_actions + self._move_actions

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]
        self.states = {}
        self.last_move = -1

    def get_availables(self):
        # 得到当前player可移动的空间
        current_chess = [move for move, p in self.states.items() if p == self.current_player]
        available_cand = list(set(range(self.width * self.height)) - set(self.states.keys()))
        availables = []
        not_enough = len(current_chess) < 3  # 棋子不够，只能下子
        if not_enough:
            acts = ["%d%d" % (self.move_to_location(move)[0], self.move_to_location(move)[1]) for move in
                    available_cand]
            availables = [i for i, key in enumerate(self.actions) if key in acts]
            availables.sort()
            return availables

        for i, action in enumerate(self.actions):
            if len(action) == 2:
                continue
            for move in current_chess:
                [h, w] = self.move_to_location(move)
                if action.startswith("%d%d" % (h, w)):
                    if self.location_to_move((int(action[2]), int(action[3]))) in available_cand:
                        availables.append(i)

        availables = list(set(availables))
        availables.sort()
        return availables

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    '''
    这里把棋盘按行编码成一个列表，move就是这个列表的索引，代表对应的棋子
    '''
    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    '''
    将动作解码成棋子的索引如何变动
    '''
    def decode_action(self, action):
        if len(action) == 2:
            # new_move, 表示下子阶段
            return self.location_to_move((int(action[0]), int(action[1]))), -1
        # new_move, old_move, 表示移子阶段
        return self.location_to_move((int(action[2]), int(action[3]))), self.location_to_move(
            (int(action[0]), int(action[1])))

    def do_move(self, action_index):
        # 先根据动作的索引，对动作进行解码
        new_move, old_move = self.decode_action(self.actions[action_index])
        self.states[new_move] = self.current_player
        if old_move >= 0:   # old_move 如果为 -1，表示是下子阶段，不需要删除
            del self.states[old_move]  # 从已经下了的棋盘上移动子
        # 切换下棋者
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = new_move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        for m in self.states.keys():
            h = m // width
            w = m % width
            player = states[m]
            # 横
            if (w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player
            # 竖
            if (h in range(height - n + 1) and len(
                    set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player
            # 右下
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            # 左下
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        return False, -1

    def get_current_player(self):
        return self.current_player

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]
