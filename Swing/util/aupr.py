__author__ = 'jfinkle'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def possible_edges(parents, children):
    """
    Create a list of all the possible edges between parents and children

    :param parents: array
        labels for parents
    :param children: array
        labels for children
    :return: array, length = parents * children
        array of parent, child combinations for all possible edges
    """
    parent_index = range(len(parents))
    child_index = range(len(children))
    a, b = np.meshgrid(parent_index, child_index)
    parent_list = parents[a.flatten()]
    child_list = children[b.flatten()]
    possible_edge_list = np.array(zip(parent_list, child_list))
    return possible_edge_list

def create_link_list(df, w):
    parent_names = df.index.values
    child_names = df.columns.values
    edges = possible_edges(parent_names, child_names)
    parents = edges[:, 0]
    children = edges[:, 1]
    directed_edges = df.values.flatten()
    all_edges = np.abs(directed_edges)
    ll_array = [parents, children, zip(parents, children), directed_edges, all_edges, w]
    link_list = pd.DataFrame(ll_array).transpose()
    link_list.columns = ['Parent', 'Child', 'Edge', 'Directed_Edge', 'Edge_Exists', 'W']
    #link_list.sort(columns='Edge_Exists', ascending=False, inplace=True)
    return link_list

def calc_pr(ref, pred):
    # True Positive Rate (TPR) = TP/(TP+FN)
    # False Positive Rate (FPR) = FP/(FP+TN)
    ref.sort(columns='Edge', inplace=True)
    pred.sort(columns='Edge', inplace=True)
    if not np.array_equal(ref.Edge.values, pred.Edge.values):
        print 'Not same edges'
        return

    pred.sort(columns='W', ascending=False, inplace=True)
    ref_edge_list = ref.Edge.values.tolist()
    num_edges = len(ref_edge_list)

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    precision = []
    recall = []

    for ii, row in enumerate(pred.iterrows()):
        pred_edge = row[1].Edge
        predicted = row[1].Edge_Exists
        ref_idx = ref_edge_list.index(pred_edge)
        real = ref.Edge_Exists.values[ref_idx]
        if real and predicted:
            tp+=1
            #print real, predicted, 'tp'
        elif not real and predicted:
            fp +=1
            #print real, predicted, 'fp'
        elif real and not predicted:
            fn +=1
            #print real, predicted, 'fn'
        elif not real and not predicted:
            tn+=1
            #print real, predicted, 'tn'
        cur_fn = num_edges-ii+1+fn
        if tp ==0 and cur_fn ==0:
            recall.append(0.0)
        else:
            recall.append(tp/(tp+cur_fn))
        if fp ==0 and tp ==0:
            precision.append(0.0)
        else:
            precision.append(tp/(tp+fp))

        aupr = integrate.cumtrapz(precision, recall)

    return precision, recall, aupr[-1]

def calc_roc(ref, pred):
    # True Positive Rate (TPR) = TP/(TP+FN)
    # False Positive Rate (FPR) = FP/(FP+TN)
    ref.sort(columns='Edge', inplace=True)
    pred.sort(columns='Edge', inplace=True)
    if not np.array_equal(ref.Edge.values, pred.Edge.values):
        print 'Not same edges'
        return

    pred.sort(columns='W', ascending=False, inplace=True)
    ref_edge_list = ref.Edge.values.tolist()
    num_edges = len(ref_edge_list)

    total_p = float(np.sum(ref.Edge_Exists.values))
    total_n = len(ref.Edge_Exists.values) - total_p

    tp = 0.0
    fp = 0.0
    tpr = []
    fpr = []

    for ii, row in enumerate(pred.iterrows()):
        pred_edge = row[1].Edge
        predicted = row[1].Edge_Exists
        ref_idx = ref_edge_list.index(pred_edge)
        real = ref.Edge_Exists.values[ref_idx]
        if real and predicted:
            tp+=1
            #print real, predicted, 'tp'
        elif not real and predicted:
            fp +=1
            #print real, predicted, 'fp'

        tpr.append(tp/total_p)
        fpr.append(fp/total_n)

        auroc = integrate.cumtrapz(fpr, tpr)
    return tpr, fpr, auroc[-1]


if __name__ == '__main__':
    xls = pd.ExcelFile('../../goldbetter_model/adjacency_matrix.xlsx')
    df = xls.parse()

    xls2 = pd.ExcelFile('../../goldbetter_model/test_matrix.xlsx')
    df2 = xls2.parse()

    np.random.seed(8)
    weights = np.random.random(len(df2)**2)
    reference = create_link_list(df, weights)
    random = np.array([np.sum(reference.Edge_Exists.values)/256.0]*256)
    prediction = create_link_list(df2, weights)
    p, r, area = calc_pr(reference, prediction)
    plt.plot(r, p, r, random, 'r')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(['Test', 'Random'])
    print area
    plt.show()

    tpr, fpr, area = calc_roc(reference, prediction)
    plt.plot(fpr, tpr, fpr, fpr, 'r')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(['Test', 'Random'], loc='best')
    print area
    plt.show()
