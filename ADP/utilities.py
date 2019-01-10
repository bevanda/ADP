from Tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for automatically plotting the gradient at the side

from typing import Dict, List, Tuple

def admissible_act(row, col, grid):
    """
    Checking possible actions in the gridworld
    and assigning it to a dict as a look-up table
    :rtype: tuple of grid_actions, state_grid

    """

    grid_actions = {}  # type: Dict[Tuple[int, int], List[str]]
    state_grid = []  # type: List[Tuple[int, int]]
    for r in range(row):
        for c in range(col):
            if not grid[r][c] == '1':
                actions = ['idle']
                if grid[r][c] != 'T' and grid[r][c] != 'G':
                    if grid[r - 1][c] != '1':
                        actions.append('up')
                    if grid[r + 1][c] != '1':
                        actions.append('down')
                    if grid[r][c - 1] != '1':
                        actions.append('left')
                    if grid[r][c + 1] != '1':
                        actions.append('right')
                grid_actions[r, c] = actions
                state_grid.append((r, c))
    return grid_actions, state_grid

def modify_state(state, row=0, col=0):
    """Function for updating a tuple values"""
    lst = list(state)
    lst[0] += row
    lst[1] += col
    return tuple(lst)

def heatmap(shape):
    """

    :type shape: tuple
    """
    row, col = shape
    uniform_data = np.random.rand(row, col)
    sns.heatmap(uniform_data, linewidth=0.5)
    plt.show()


def plots():
    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    return 0



def print_dec(original_func):
    def wrapper_func(*args, **kwargs):
        print "Grid size"
        print original_func(*args, **kwargs)
        return original_func(*args, **kwargs)

    return wrapper_func


class DecoratorClass(object):

    def __init__(self, original_func):
        self.original_func = original_func

    def __call__(self, *args, **kwargs):
        print('call method executed before {}'.format(self.original_func.__name__))
