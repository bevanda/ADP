# here is the main for running the algorithm
from typing import Tuple

from ADP.environments import *

from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt GridWorld into the
import sys


if __name__ == "__main__":

    World = GridWorld()
    World.open_world()
    World.print_grid()
    state_action = World.allowed_actions()
    state = World.start_pos()
    print state
    print World.next_state(state, 'down')

