__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import os
import sys
import time
import pandas as pd
import numpy as np
from Evaluator import Evaluator
from sklearn.neighbors import DistanceMetric
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as sch
from scipy.stats import pearsonr
from scipy import stats
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import KMeans
from gmpy2 import is_square

def load_roller_pickles(pickle_path):
    '''
    Load pickles into an analyzable data structures
    :param pickle_path:
    :return:
    '''
    file_list = next(os.walk(pickle_path))[2]
    nfiles = len(file_list)

    obj_list = []
    counter = 0

    dataset_dict = {}

    #Add the roller objects to the dataset key
    for filename in file_list:
      current_file_path = pickle_path + filename
      roller_obj = pd.read_pickle(current_file_path)
      attributes = dir(roller_obj)
      if any("file_path" in attribute for attribute in attributes):
        counter += 1
        #print(str(counter) + " out of " + str(nfiles))
        obj_list.append(roller_obj)

        dataset_key = roller_obj.file_path
        if dataset_key not in dataset_dict:
          dataset_dict[dataset_key] = {'roller_list': [], 'results_frame':None, 'canberra_distance': None, 'auroc_difference': None}

        dataset_dict[dataset_key]['roller_list'].append(roller_obj)

    #Compile results for all rollers of a given dataset into a dataframe
    for key, value in dataset_dict.iteritems():
        current_results = make_window_table(value['roller_list'])
        value['results_frame'] = current_results
        value['canberra_distance'] = calc_canberra(current_results)
        value['auroc_difference'] = calc_auroc_diff(current_results)
    return dataset_dict, obj_list

def make_window_table(roller_list):
    df = pd.DataFrame(columns=['Timeseries', 'Width', 'Roller_idx', 'Start', 'Goldstandard', 'AUROC','Edges', 'Edge_ranking'])
    count = 0
    for roller in roller_list:
        current_timeseries = roller.file_path
        current_gold_standard = current_timeseries.replace("timeseries.tsv","goldstandard.tsv")
        current_gold_standard = '../../'+current_gold_standard
        current_width = roller.window_width
        evaluator = Evaluator(current_gold_standard, '\t')
        for idx, window in enumerate(roller.window_list):
            start_time = min(window.raw_data['Time'].unique())
            unsorted = window.results_table[np.isfinite(window.results_table['p_value'])]
            edge_sorted = unsorted.sort(['regulator-target'])
            edge_list = edge_sorted['regulator-target'].values
            current_ranking = edge_sorted['p_value-rank'].values.astype(int)
            current_ranking = tuple(max(current_ranking)-current_ranking+1)
            sorted = unsorted.sort(['p_value'], ascending=[True])
            #print sorted
            importance = unsorted.sort(['importance', 'p_value'], ascending=[False, True])

            #print current_ranking

            tpr, fpr, auroc = evaluator.calc_roc(sorted)
            #print auroc[-1]
            #print filtered.shape
            tpr, fpr, auroc = evaluator.calc_roc(importance)

            #print auroc[-1]
            df.loc[count] = [current_timeseries, current_width, idx, start_time, current_gold_standard, auroc[-1],
                             edge_list, current_ranking]
            count+=1
    return df

def calc_canberra(data_frame):
    rankings = np.array([list(ranking) for ranking in data_frame['Edge_ranking'].values])
    dist = DistanceMetric.get_metric('canberra')
    can_dist = dist.pairwise(rankings)
    return can_dist

def calc_auroc_diff(data_frame):
    aurocs = data_frame['AUROC'].values
    indices = range(len(aurocs))
    col_idx, row_idx = np.meshgrid(indices, indices)
    auroc_mat = np.abs(aurocs[row_idx] - aurocs[col_idx])
    return auroc_mat

def basic_heatmap(data, show_now=True):
    plt.figure()
    plt.pcolor(data, cmap=cm.RdBu)
    plt.colorbar()
    plt.xlim([0, data.shape[1]])
    if show_now:
        plt.show()

def visualize_raw_data(roller_obj):
    roller_data = roller_obj.raw_data
    x = np.arange(0, len(roller_data.columns.values))
    y = roller_data.Time.values
    z = roller_data.values[:, 1:]
    n_genes = z.shape[1]
    times = np.sort(list(set(y)))
    n_samples = len(np.where(y==times[0])[0])
    n_times = len(times)
    data = z[:n_times, :]
    for ii in range(2, n_samples+1):
        data = np.hstack((data, z[n_times*(ii-1):n_times*ii, :]))
    column_order = np.array([np.arange(0, n_genes*n_samples,n_genes)+jj for jj in range(n_genes)]).flatten()
    data = data[:, column_order].T

    data_diff = np.diff(data)
    time_diff = np.diff(times)
    rates = data_diff/time_diff

    basic_heatmap(data)
    basic_heatmap(rates)

def show_window_results(data_frame, org='max_auroc'):
    current_frame = data_frame['results_frame']

    sort_indices = np.argsort(current_frame['AUROC'].values)[::-1]
    sorted_auroc = current_frame['AUROC'].values[sort_indices]
    sorted_auroc_difference = data_frame['auroc_difference'][sort_indices]
    sorted_canberra = data_frame['canberra_distance'][sort_indices]

    # The first row now corresponds to the window with the highest AUROC.
    # Get the indices for ascending canberra distance
    new_sort = np.argsort(sorted_canberra[0])
    sort_indices = new_sort
    rankings = np.array([list(ranking) for ranking in current_frame['Edge_ranking'].values])

    sorted_rankings = rankings[sort_indices]

    # Compute and plot dendrogram.
    fig = plt.figure()
    axdendro = fig.add_axes([0.06, 0.01, 0.9, 0.05])
    Y = sch.linkage(sorted_rankings, metric='canberra')
    Z = sch.dendrogram(Y, orientation='bottom')
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis("off")

    # Plot distance matrix.
    #axmatrix = fig.add_axes([dm_left, dm_bottom, dm_width, dm_height])

    axmatrix = fig.add_axes([0.06, 0.07, 0.9, 0.5])
    if org == 'max_auroc':
        index = new_sort
    else:
        index = Z['leaves']

    sorted_auroc = current_frame['AUROC'].values[index]

    D_ordered = rankings[index] # Reorder
    im = axmatrix.matshow(D_ordered.T, aspect='auto')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    im.set_cmap(cm.Blues_r)

    #Colorbar
    axcolor = fig.add_axes([0.02, 0.07, 0.02, 0.5])
    cbar=plt.colorbar(im, cax=axcolor, orientation='vertical')
    axcolor.tick_params(labelsize=12, labeltop=True, labelbottom=True)

    auc_ax = fig.add_axes([0.06, 0.6, 0.9, 0.1])
    auc_diff = sorted_auroc-max(sorted_auroc)
    auc_ax.bar(range(len(sorted_auroc)), auc_diff, color='b', width=1, align='center')
    auc_ax.set_xlim([0, len(sorted_auroc)])
    auc_ax.set_ylim([min(auc_diff), abs(min(auc_diff))])
    can_ax = auc_ax.twinx()
    current_can = sorted_canberra[0][new_sort]
    can_ax.bar(range(len(sorted_auroc)), current_can, color='r', width=1, align='center')
    can_ax.set_ylim([-max(current_can), max(current_can)])
    can_ax.set_xlim([0, len(sorted_auroc)])

    width_ax = fig.add_axes([0.06, 0.73, 0.9, 0.1])
    sorted_w = current_frame['Width'].values[index]
    width_ax.bar(range(len(sorted_w)), sorted_w, width=1)
    width_ax.set_xlim([0, len(sorted_auroc)])
    width_ax.set_ylim([min(sorted_w), max(sorted_w)])

    start_ax = fig.add_axes([0.06, 0.86, 0.9, 0.1])
    sorted_start = current_frame['Start'].values[index]
    start_ax.bar(range(len(sorted_start)), sorted_start, width=1)
    start_ax.set_xlim([0, len(sorted_start)])

    plt.show()

def explore_rankings(data_frame):
    '''
    The goal here is to see which windows need to be combined to get perfect network reconstruction...
    if that is even possible

    :param data_frame:
    :return:
    '''
    sort_indices = np.argsort(data_frame['AUROC'].values)[::-1]
    rankings = np.array([list(ranking) for ranking in data_frame['Edge_ranking'].values])
    sorted_rankings = rankings[sort_indices]

    edge_list = data_frame['Edges'][0].tolist()
    evaluator = Evaluator(gs, '\t')
    true_edges = evaluator.gs_flat.tolist()
    n_true = len(true_edges)
    print n_true
    if is_square(n_true):
        rows = np.sqrt(n_true)
        columns = np.sqrt(n_true)
    else:
        rows = np.floor(np.sqrt(n_true))+1
        columns = np.floor(np.sqrt(n_true))+1

    #Let's see how well each edge gets predicted
    #hist_fig = plt.figure()
    rank_cutoff = n_true
    success_dict = {}
    for ii,edge in enumerate(true_edges):
        current_plot = plt.subplot(rows, columns, ii+1)
        edge_idx = edge_list.index(edge)
        rows_for_dict = data_frame[rankings[:, edge_idx]<=n_true].iloc[:, [0,1,3,5,7]]
        #current_plot.hist(rankings[:, edge_idx])
        #current_plot.set_title(edge)
        success_dict[edge] = rows_for_dict
        plt.subplot(3,1, 1)
        plt.hist(rows_for_dict.Width.values)
        plt.subplot(3,1, 2)
        plt.hist(rows_for_dict.Start.values)
        plt.subplot(3,1, 3)
        plt.hist(rows_for_dict.AUROC.values)
        plt.show()

    #plt.show()    
if __name__ == '__main__':
    #path = "../../output/Roller_outputs_RF_moretrees/"
    #roller_dict, roller_list = load_roller_pickles(path)
    #pd.to_pickle(roller_dict, "../../output/results_pickles/Roller_outputs_RF.pickle")

    #roller_dict = pd.read_pickle("../../output/results_pickles/Roller_outputs_RF.pickle")
    #roller_dict = pd.read_pickle("../../output/results_pickles/Roller_outputs_RF_moretrees.pickle")

    #print len(roller_dict.keys())
    window_size_array = np.load("../../output/results_pickles/window_size.npy")
    best_scores_array = np.load("../../output/results_pickles/best_scores.npy")
    with PdfPages("../../output/results_pickles/figure.pdf") as pdf:
        fig = plt.figure(figsize=(11,9))
        ax = fig.add_subplot(111)
        ax.plot(window_size_array, best_scores_array, 'o-', linewidth=5, markersize=10)
        plt.xlabel('Test', size=20, style='italic')
        plt.ylabel('Ylabel', size=20)
        plt.xlim([1,22])
        plt.grid(color='k', linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        legend_names = ['in silico network 1', 'in silico network 2', 'in silico network 3', 'in silico network 4', 'in silico network 5']
        plt.legend(legend_names, bbox_to_anchor=(-0.1, 0.55, 1., .102))

        pdf.savefig()
        plt.close()

    sys.exit()
    for dataset, df in roller_dict.iteritems():
        print dataset
        gs = '../../' + dataset.replace("timeseries.tsv","goldstandard.tsv")
        current_frame = df['results_frame']

        window_size_list = current_frame.Width.values
        window_score_list = current_frame.AUROC.values
        window_size_set = np.array(list(set(window_size_list)))
        best_scores = np.array([max(window_score_list[window_size_list==size]) for size in window_size_set])
        if window_size_array is None:
            window_size_array = window_size_set
        else:
            window_size_array = np.vstack((window_size_array, window_size_set))
        if best_scores_array is None:
            best_scores_array = best_scores
        else:
            best_scores_array = np.vstack((best_scores_array, best_scores))

        #explore_rankings(current_frame)
        #visualize_raw_data(df['roller_list'][0])
        #show_window_results(df)

        #sys.exit()

