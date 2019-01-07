import sys
from ADP.utilities import *
from Tkinter import *


class GridWorld(object):
    """Reading the path to the .txt file describing the GridWorld

    1 - wall
    0 - free
    G - goal
    T - trap
    S - start

    """

    def __init__(self):

        self.col = 0
        self.row = 0
        self.state_action = {}  # dictionary as a look-up table
        #self.succ_state = {}
        self.grid_world = []
        self.actions=['up', 'down', 'left', 'right', 'idle']
        self.program_name = sys.argv[0]
        self.arguments = sys.argv[1:]
        self.count = len(self.arguments)
        if len(sys.argv) != 2:
            self.file_name = "/home/petar/test_maze.txt"
            #raise Exception("Need two arguments: arg1:=script_name  arg2:=path_to_the_file!")
        else:
            self.file_name = self.arguments[0]

    def open_world(self):
        with open(self.file_name, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln.startswith("#"):
                    ln = ln.split()
                    self.grid_world.append(ln)

        return self.grid_world

    def get_world(self):
        return self.file_name

    def set_world(self, new_path):
        self.file_name = new_path

    def print_grid(self):
        print('\n'.join(map(' '.join, self.grid_world)))

    def run_gui(self):
        root = Tk()
        GUI(root)
        root.mainloop()

    @print_dec
    def get_size(self):
        self.col = len(self.grid_world[0])
        self.row = len(self.grid_world)
        return [self.row, self.col]

    def allowed_actions(self):
        """looking at 'S's and '0's """
        self.get_size()
        self.state_action = admissible_act(self.row, self.col, self.grid_world)
        return self.state_action

    def next_state(self, cur_state, action):
        # @state is a tuple
        i = 0
        j = 0
        self.allowed_actions()
        acts = self.state_action.get(cur_state)
        if action in acts:
            if action == 'up':
                i = -1
            elif action == 'down':
                i = +1
            elif action == 'left':
                j = -1
            elif action == 'right':
                j = 1
        else:
            print "No move made"
        return modify_state(cur_state, i, j)

    def get_actions(self):
        return self.actions

    def set_actions(self, actions):
        self.actions = actions

    def start_pos(self):
        for r in range(self.row):
            for c in range(self.col):
                if self.grid_world[r][c] == 'S':
                    return r, c

