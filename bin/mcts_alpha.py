'''
根据蒙特卡罗搜索树获取当前状态下，可走动作的概率分布，根据策略选择下一步动作。

- 蒙特卡罗搜索树
	```
	1. 随机生成多幕数据
	2. 每幕对战数据知道终局
	3. 策略价值网络评估终局价值
	4. 价值反传，更新蒙特卡罗搜索树
	5. 根据UCB得到当前状态下各个可行动作的概率分布
	```
- 策略选择动作
	根据动作的概率分布随机选择，或者加入一定噪声做探索。
'''

# -*- coding: utf-8 -*-

import copy

import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """每一次探索，都从根节点到叶子节点，然后评估叶子节点的策略价值，再将策略价值反向反馈回去。
        经过多轮迭代后，就能知道哪条路径的策略价值更高
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # 模拟阶段，对战双方只能贪婪的选择概率最大的单次移动
            # 返回动作的索引，和对应的节点
            action_index, node = node.select(self._c_puct)
            # 让棋盘模拟走子
            state.do_move(action_index)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 这里评估的是有效动作的概率
        action_probs, leaf_value = self._policy(state)
        # 检测游戏是否达到结束的条件
        end, winner = state.game_end()
        if not end:
            # 没有结束就需要更新蒙特卡洛树，添加下一步的节点
            node.expand(action_probs)
        else:
            # 如果平局，策略价值为 0
            if winner == -1:  # tie
                leaf_value = 0.0
            else: # 赢了，策略价值为 1; 输了，策略价值为 -1
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)

        # 在当前探索中，递归的更新每个节点的策略价值和访问次数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state，棋盘状态
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            # 蒙特卡洛树会重用，每一次移动都会改变，所以要先保存起来，每次都能从当前棋盘状态开始探索
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        # 探索完成后，当前蒙特卡洛树记录的是胜出的路径形成的最大可能的概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        action_probs = np.zeros(len(board.actions))  # 存储动作空间中每个动作的概率
        acts, probs = self.mcts.get_move_probs(board, temp)  # 这里的 acts 为动作的索引
        action_probs[list(acts)] = probs    # 每个动作的概率
        if self._is_selfplay:
            # 添加了狄利克雷噪音的探索
            action = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            # update the root node and reuse the search tree
            self.mcts.update_with_move(action)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            # 按概率分布选择一个动作
            action = np.random.choice(acts, p=probs)
            # reset the root node
            self.mcts.update_with_move(-1)

        if return_prob:
            return action, action_probs
        else:
            return action

    def __str__(self):
        return "MCTS {}".format(self.player)
