# here is the main for running the algorithm
from ADP.environments import *

from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt GridWorld into the
import sys


if __name__ == "__main__":

    World = GridWorld()
    Grid = World.open_world()
    Size = World.get_size()
    World.print_grid()
    print Grid[3][8]


