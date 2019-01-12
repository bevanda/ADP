from Tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for automatically plotting the gradient at the side
from pprint import pprint
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


# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw={}, cbarlabel="", **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.
#
#     Arguments:
#         data       : A 2D numpy array of shape (N,M)
#         row_labels : A list or array of length N with the labels
#                      for the rows
#         col_labels : A list or array of length M with the labels
#                      for the columns
#     Optional arguments:
#         ax         : A matplotlib.axes.Axes instance to which the heatmap
#                      is plotted. If not provided, use current axes or
#                      create a new one.
#         cbar_kw    : A dictionary with arguments to
#                      :meth:`matplotlib.Figure.colorbar`.
#         cbarlabel  : The label for the colorbar
#     All other arguments are directly passed on to the imshow call.
#     """
#
#     if not ax:
#         ax = plt.gca()
#
#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)
#
#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
#
#     # We want to show all ticks...
#     ax.set_xticks(np.arange(data.shape[1]))
#     ax.set_yticks(np.arange(data.shape[0]))
#     # ... and label them with the respective list entries.
#     ax.set_xticklabels(col_labels)
#     ax.set_yticklabels(row_labels)
#
#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#              rotation_mode="anchor")
#
#     # Turn spines off and create white grid.
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)
#
#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     return im, cbar
#
#
# def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
#                      textcolors=["black", "white"],
#                      threshold=None, **textkw):
#     """
#     A function to annotate a heatmap.
#
#     Arguments:
#         im         : The AxesImage to be labeled.
#     Optional arguments:
#         data       : Data used to annotate. If None, the image's data is used.
#         valfmt     : The format of the annotations inside the heatmap.
#                      This should either use the string format method, e.g.
#                      "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
#         textcolors : A list or array of two color specifications. The first is
#                      used for values below a threshold, the second for those
#                      above.
#         threshold  : Value in data units according to which the colors from
#                      textcolors are applied. If None (the default) uses the
#                      middle of the colormap as separation.
#
#     Further arguments are passed on to the created text labels.
#     """
#
#     if not isinstance(data, (list, np.ndarray)):
#         data = im.get_array()
#
#     # Normalize the threshold to the images color range.
#     if threshold is not None:
#         threshold = im.norm(threshold)
#     else:
#         threshold = im.norm(data.max())/2.
#
#     # Set default alignment to center, but allow it to be
#     # overwritten by textkw.
#     kw = dict(horizontalalignment="center",
#               verticalalignment="center")
#     kw.update(textkw)
#
#     # Get the formatter in case a string is supplied
#     if isinstance(valfmt, str):
#         valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
#
#     # Loop over the data and create a `Text` for each "pixel".
#     # Change the text's color depending on the data.
#     texts = []
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
#             text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
#             texts.append(text)
#
#     return texts


def heatmap(env, V):
    """

    """

    sns.color_palette("bright")
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")


    data = np.zeros(shape=env.shape)
    data.fill(np.nan)
    for i in range(len(V)):
        r, c = env.subs2idx(i)
        data[r][c] = V[i]
    # #data = np.flipud(data)
    # data = np.fliplr(data)
    # dy, dx = np.gradient(data)
    # plt.imshow(data, origin='lower')
    # pprint(data)
    # plt.quiver(dx, dy)
    ax = sns.heatmap(data, linewidth=0.1)
    # no line to numbers
    ax.tick_params(length=0)
    #xlabel on top
    plt.yticks(rotation=0)
    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
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
