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

        self.row = 0
        self.col = 0
        self.state_action={} #dictionary as a look-up table
        self.grid_world = []
        self.program_name = sys.argv[0]
        self.arguments = sys.argv[1:]
        self.count = len(self.arguments)
        if len(sys.argv) != 2:
            raise Exception("Need two arguments: arg1:=script_name  arg2:=path_to_the_file!")
        self.file_name = self.arguments[0]

    def open_world(self):
        with open(self.file_name, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln.startswith("#"):
                    ln=ln.split()
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
        self.row = len(self.grid_world[0])
        self.col = len(self.grid_world)
        return [self.row, self.col]


    def allowed_actions(self):
        """looking at 'S's and '0's """
        self.get_size()
        for i in range(self.row):
            for j in range(self.col):




