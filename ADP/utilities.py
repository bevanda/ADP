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

def check(x, y, grid):
    """Checking the type of square in the gridworld
    grid - nested list
    """

    print 'visiting %d,%d' % (x, y)

    if grid[x][y] == '0':
        print 'Free at %d,%d' % (x, y)

    elif grid[x][y] == 'S':
        print 'Start at %d,%d' % (x, y)

    else:
        print "No action possible from there"

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
