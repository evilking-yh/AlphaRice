# -*- coding: utf-8 -*-
from __future__ import print_function

from bin.board import Board
from bin.game import Game
from bin.mcts_alpha import MCTSPlayer
from bin.policy_value_net import PolicyValueNet


class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        action = None
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                items = location.split('->')
                if len(items) == 1:
                    location = [int(n, 10) for n in items[0].split(",")]
                    action = "%d%d" % (location[0], location[1])
                elif len(items) == 2:
                    new_location = [int(n, 10) for n in items[1].split(",")]
                    old_location = [int(n, 10) for n in items[0].split(",")]
                    action = "%d%d%d%d" % (old_location[0], old_location[1], new_location[0], new_location[1])
        except Exception as e:
            action = None
        if action is None:
            print("invalid move")
            action = self.get_action(board)
        return board.actions.index(action)

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 3
    width, height = 3, 3
    model_file = 'current_policy.model'
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    best_policy = PolicyValueNet(width, height, len(board.actions), model_file=model_file)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
    human = Human()

    # set start_player=0 for human first
    game.start_play(human, mcts_player, start_player=0, is_shown=1)


if __name__ == '__main__':
    run()
