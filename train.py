# -*- coding: utf-8 -*-
from __future__ import print_function

import random
from collections import deque

import numpy as np

from bin.board import Board
from bin.game import Game
from bin.mcts_alpha import MCTSPlayer
from bin.policy_value_net import PolicyValueNet


class TrainPipeline():
    def __init__(self, init_model=None):
        # 棋盘的大小设置
        self.board_width = 3
        self.board_height = 3
        self.n_in_row = 3   # 多少颗子连成一条线，米字棋是三颗子连成一条线即为胜利
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # 每一次移动模拟多少次play
        self.c_puct = 5
        self.buffer_size = 10000    # 收集多少对战数据后开始训练策略评估函数
        self.batch_size = 64  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 2000  # 多少轮 selfplay
        self.best_win_ratio = 0.0
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height,
                                                   len(self.board.actions), model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height,
                                                   len(self.board.actions))
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def collect_selfplay_data(self, n_games=1):
        """收集 self-play 对战数据"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # 判断策略价值网络更新前与更新后的概率分布的相似度，如果比较相似了，即KL散度比较大，说明收敛了，早停
            # KL散度越大，训练越接近尾声
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            # adaptively adjust the learning rate
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

    def run(self):
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
            if len(self.data_buffer) > self.batch_size:
                self.policy_update()

            if (i + 1) % self.check_freq == 0:
                print("current self-play batch: {}".format(i + 1))
                self.policy_value_net.save_model('./current_policy.model')

        self.policy_value_net.save_model('./last_policy.model')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
