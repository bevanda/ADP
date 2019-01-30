import time
import numpy as np
import matplotlib.pyplot as plt

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
                if grid[r][c] != 'G':
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


def visualise(env, V, policy):
    """
    Visualising the policy and the underlying cost heat map
    """
    # data = np.zeros(shape=env.shape)
    data = np.full(env.shape, np.nan)
    x = []
    y = []
    dx = []
    dy = []
    # plt.figure(figsize=(4, 8))
    for i in range(len(V)):
        r, c = env.subs2idx(i)
        # - (np.max(V)+1) for normalising the grid
        # data[r][c] = V[i] - (np.max(V)+10)
        data[r][c] = V[i]
        x.append(c)
        y.append(r)
        an = np.argmax(policy[i])
        action = env.action_list[an]
        if action == 'up':
            dx.append(0)
            dy.append(-0.5)
        elif action == 'down':
            dx.append(0)
            dy.append(0.5)
        elif action == 'left':
            dx.append(-0.5)
            dy.append(0)
        elif action == 'right':
            dx.append(0.5)
            dy.append(0)
        elif action == 'idle':
            dx.append(0)
            dy.append(0)
    print V

    plt.set_cmap('RdYlBu_r')
    data = np.ma.masked_where(np.isnan(data), data)
    cmap = plt.cm.get_cmap()
    cmap.set_bad(color='black')
    costs = plt.imshow(data,
                       # origin='lower'
                       )
    # print data
    v = np.linspace(min(V), max(V), endpoint=True)
    plt.colorbar(costs,
                 # ticks=v
                 )

    plt.quiver(x, y, dx, dy,
               angles='xy', scale_units='xy', scale=1.)

    return plt


def timeit(method):
    """Decorator for timing a code snippet"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result

    return timed
