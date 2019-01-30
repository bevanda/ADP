import sys
from typing import Dict, List, Tuple, Any
import random
from ADP.utilities import *


class Maze(object):
    """Reading the path to the .txt file describing the Maze

    1 - wall
    0 - free
    G - goal
    T - trap
    S - start

    """

    def __init__(self, g=1):

        self.program_name = sys.argv[0]
        self.arguments = sys.argv[1:]
        self.count = len(self.arguments)

        if len(sys.argv) != 2:
            # self.file_name = "/home/petar/test_maze.txt"
            raise Exception("Need absolute path to file as argument!")
        else:
            self.file_name = self.arguments[0]

        self.grid_world = []  # type: List[Any]
        self.action_list = \
            [
                'up',
                'down',
                'left',
                'right',
                'idle'
            ]
        self.state_num = None
        self.act_num = None
        self.shape = None  # type: Tuple[int, int]
        self.grid_actions = None  # type: (Dict[Tuple[int, int], List[str]])
        self.state_grid = None  # type: (List[Tuple[int, int]])
        self._current_state = None  # type: Tuple[int, int]
        self.P_g = {}
        self.cost = []
        self.col = 0
        self.row = 0
        self.start_state = 0

        self._open_world()
        self.print_grid()
        self._get_shape()
        self.start_pos()
        self._allowed_actions()
        self.MRP(g)
        # self.probability_transitions()

    def _open_world(self):
        with open(self.file_name, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln.startswith("#"):
                    ln = ln.split()
                    self.grid_world.append(ln)

    def get_file_path(self):
        return self.file_name

    def set_file_path(self, new_path):
        self.file_name = new_path

    def start_pos(self):
        for r in range(self.row):
            for c in range(self.col):
                if self.grid_world[r][c] == 'S':
                    self.current_state = r, c
                    return r, c

    @property
    def num_states(self):
        self.state_num = len(self.state_grid)
        return self.state_num

    @property
    def num_actions(self):
        self.act_num = len(self.actions)
        return self.act_num

    def print_grid(self):
        print('\n'.join(map(' '.join, self.grid_world)))

    def _get_shape(self):
        self.col = len(self.grid_world[0])
        self.row = len(self.grid_world)
        self.shape = self.row, self.col
        return self.shape  # returns a tuple just as np.shape for np arrays

    def _allowed_actions(self):
        """looking at 'S's and '0's """
        self.grid_actions, self.state_grid = admissible_act(self.row, self.col, self.grid_world)

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state_tuple):
        self._current_state = state_tuple

    def next_state(self, state, action):
        """
        :param action: the action to take in the environment
        :type action: string

        :param state: the position of the agent in the maze (row, col)
        :type state: tuple
        """
        i = 0
        j = 0
        acts = self.grid_actions[state]
        if action in acts:
            if action == 'up':
                i = -1
            elif action == 'down':
                i = +1
            elif action == 'left':
                j = -1
            elif action == 'right':
                j = 1
            # elif action == 'idle':
        else:
            pass
            # print "No move made"
        return modify_state(state, i, j)

    def possible_actions(self, subs):
        return self.grid_actions[self.subs2idx(subs)]

    def subs2idx(self, subs):
        return self.state_grid[subs]

    def idx2subs(self, idx):
        return self.state_grid.index(idx)

    @property
    def actions(self):
        return self.action_list

    @actions.setter
    def actions(self, act):
        self.action_list = act

    def back2start(self):
        self.current_state = self.start_pos()

    def MRP(self, g):
        for s in range(self.num_states):
            p = {}
            current_state = s
            for _, act in enumerate(self.possible_actions(s)):
                an = self.action_list.index(act)
                p[an] = self.action_probability(current_state, an, g)
            self.P_g[s] = p
        return self.P_g

    def stage_cost(self, state_no, next_state_no, g):
        cS_r, cS_c = self.subs2idx(state_no)
        nS_r, nS_c = self.subs2idx(next_state_no)
        if g == 2:
            if self.grid_world[cS_r][cS_c] == 'G' and self.grid_world[nS_r][nS_c] == 'G':
                cost = 0
            elif self.grid_world[nS_r][nS_c] == 'T':
                cost = 50
            else:
                cost = 1
        else:
            if self.grid_world[nS_r][nS_c] == 'G':
                if self.grid_world[cS_r][cS_c] == 'G':
                    cost = 0
                else:
                    cost = -1
            elif self.grid_world[nS_r][nS_c] == 'T':
                cost = 50
            else:
                cost = 0

        return cost

    def action_probability(self, state_no, action_no, g, p=0.1):
        pg = []
        nxt_state = self.next_state(self.subs2idx(state_no), self.action_list[action_no])
        next_state_no = self.idx2subs(nxt_state)
        action_available = self.grid_actions[nxt_state]
        action = self.action_list[action_no]
        if action == 'up' or action == 'down':
            if 'left' in action_available and 'right' in action_available:
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((1 - 2 * p, next_state_no, cost))
                # go diagonally left
                next_state_no = self.idx2subs(self.next_state(nxt_state, 'left'))
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((p, next_state_no, cost))
                # go diagonally right
                next_state_no = self.idx2subs(self.next_state(nxt_state, 'right'))
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((p, next_state_no, cost))
            else:
                if 'left' in action_available:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1 - p, next_state_no, cost))
                    # go diagonally left
                    next_state_no = self.idx2subs(self.next_state(nxt_state, 'left'))
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((p, next_state_no, cost))
                elif 'right' in action_available:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1 - p, next_state_no, cost))
                    # go diagonally left
                    next_state_no = self.idx2subs(self.next_state(nxt_state, 'right'))
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((p, next_state_no, cost))
                else:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1, next_state_no, cost))
        elif action == 'left' or action == 'right':
            if 'up' in action_available and 'down' in action_available:
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((1 - 2 * p, next_state_no, cost))
                # go diagonally left
                next_state_no = self.idx2subs(self.next_state(nxt_state, 'up'))
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((p, next_state_no, cost))
                # go diagonally right
                next_state_no = self.idx2subs(self.next_state(nxt_state, 'down'))
                cost = self.stage_cost(state_no, next_state_no, g)
                pg.append((p, next_state_no, cost))
            else:
                if 'up' in action_available:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1 - p, next_state_no, cost))
                    # go diagonally left
                    next_state_no = self.idx2subs(self.next_state(nxt_state, 'up'))
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((p, next_state_no, cost))
                elif 'down' in action_available:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1 - p, next_state_no, cost))
                    # go diagonally left
                    next_state_no = self.idx2subs(self.next_state(nxt_state, 'down'))
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((p, next_state_no, cost))
                else:
                    cost = self.stage_cost(state_no, next_state_no, g)
                    pg.append((1, next_state_no, cost))
        else:
            cost = self.stage_cost(state_no, next_state_no, g)
            pg.append((1, next_state_no, cost))
        return pg
