from Tkinter import *
import numpy as np
import matplotlib.pyplot as plt

class GUI(object):

    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

def admissible_act(row, col, grid):
    """Checking possible actions in the gridworld
    and assigning it to a dict as a look-up table
    """
    state_action={}
    for r in range(row):
        for c in range(col):
            if not grid[r][c] == '1':
                actions = ['idle']
                if grid[r][c] != 'T' and grid[r][c] != 'G':
                    if grid[r-1][c] != '1':
                        actions.append('up')
                    if grid[r+1][c] != '1':
                        actions.append('down')
                    if grid[r][c-1] != '1':
                        actions.append('left')
                    if grid[r][c+1] != '1':
                        actions.append('right')
                state_action[r, c] = actions
    return state_action

def modify_state(state, row = 0, col = 0):
    lst = list(state)
    lst[0] += row
    lst[1] += col
    return tuple(lst)

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
        self.original_func=original_func

    def __call__(self, *args, **kwargs):
        print('call method executed before {}'.format(self.original_func.__name__))
