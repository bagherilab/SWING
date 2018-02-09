import pandas as pd
import scipy.stats as ss
import sklearn.metrics as skmet
import numpy as np

def make_possible_edge_list(parents, children, self_edges=True):
    """
    Create a list of all the possible edges between parents and children

    :param parents: array
        labels for parents
    :param children: array
        labels for children
    :param self_edges:
    :return: array, length = parents * children
        array of parent, child combinations for all possible edges
    """
    parent_index = range(len(parents))
    child_index = range(len(children))

    a, b = np.meshgrid(parent_index, child_index)
    parent_list = list(parents[a.flatten()])
    child_list = list(children[b.flatten()])
    possible_edge_list = None
    if self_edges:
        possible_edge_list = list(zip(parent_list, child_list))

    elif not self_edges:
        possible_edge_list = [x for x in zip(parent_list, child_list) if x[0] != x[1]]

    return possible_edge_list