import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import pdb
import numpy as np

#this needs to be refactored to be an actual class.

def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_lines(series_list, regulator_labels, target_labels, window_size, suffix=""):
    figure2 = plt.figure()
    lineplot = figure2.add_subplot(1,1,1)
    lineplot.set_xlabel('start day')
    lineplot.set_ylabel('Beta')
    lines = []
    time = [x for x in range(0,22-window_size)]
    label_list = []
    for counter,series in enumerate(series_list):
        my_label = str(regulator_labels[counter]+" -> "+target_labels[counter])
        label_list.append(my_label)
        line, = lineplot.plot(time, series, label = my_label)
        lines.append(line)
    figure2.legend(lines,label_list)
    figure2.savefig('line_figure'+str(window_size)+suffix+'.png')


#generating a heatmap figure
def generate_heatmap_from_df(raw_data):
    """ returns heatmap figure and axes """
    time_vector = raw_data['Time'].unique()
    nrow,ncol=raw_data.shape
    nrepeats = nrow/len(time_vector)
    #create group assignment
    groups = [x for x in range(0,nrepeats) for i in range(0,len(time_vector))]
    raw_data['Group'] = groups
    sorted = raw_data.sort(['Time','Group'])
    heatmap_values = pd.DataFrame()
    col_labels = map(int,time_vector)
    partial_row_labels = raw_data.drop(['Time','Group'],1).columns.values.tolist()
    row_labels = [i for x in range(0,nrepeats) for i in partial_row_labels]
    for time in time_vector: 
        target_set = sorted[sorted['Time']==time].drop(['Time','Group'],1)
        vector = target_set.values.flatten()
        heatmap_values[time] = vector
    figure1 = plt.figure()
    heatmap = figure1.add_subplot(2,1,1)
    my_axis = heatmap.imshow(heatmap_values,interpolation='nearest',cmap=cm.RdBu)
    my_axi = my_axis.get_axes()
    clean_axis(my_axi)
    heatmap.set_yticks(np.arange(heatmap_values.shape[0]))
    heatmap.yaxis.set_ticks_position('left')
    heatmap.set_yticklabels(row_labels)
    heatmap.set_xticks(np.arange(heatmap_values.shape[1]))
    heatmap.xaxis.set_ticks_position('top')
    xlabelsL = heatmap.set_xticklabels(col_labels)
    for label in xlabelsL:
        label.set_rotation(90)
        label.set_fontsize(8)
    for label in (heatmap.get_yticklabels()):
        label.set_fontsize(8)
    for l in heatmap.get_xticklines() + heatmap.get_yticklines():
        l.set_markersize(0)
    title=heatmap.set_title("Heatmap")
    
    
    title.set_x(1.2)
    figure1.savefig("/home/jjw036/Swing/yeast_heatmap.png")
    return(figure1, heatmap)




#order the raw data such that it is g1 sample1 sample2 sample3 ... g2 etc.


def plot_figure(coeff_matrix,nth_window, row_labels, col_labels, window_size,prefix=""):
    df = pd.DataFrame(coeff_matrix)
    figure1 = plt.figure()
    heatmap = figure1.add_subplot(1,1,1)
    my_axis = heatmap.imshow(df,interpolation='nearest',cmap=cm.OrRd, vmin=0, vmax=1.2)
    my_axi = my_axis.get_axes()
    clean_axis(my_axi)
    heatmap.set_yticks(np.arange(df.shape[0]))
    heatmap.yaxis.set_ticks_position('left')
    heatmap.set_yticklabels(row_labels)
    heatmap.set_xticks(np.arange(df.shape[1]))
    heatmap.xaxis.set_ticks_position('top')
    xlabelsL = heatmap.set_xticklabels(col_labels)
    for label in xlabelsL:
        label.set_rotation(90)
    for l in heatmap.get_xticklines() + heatmap.get_yticklines():
        l.set_markersize(0)
    title=heatmap.set_title("Day " +str(nth_window) + " to Day " +str(nth_window+window_size))
    title.set_x(1.2)
    figure1.savefig(prefix+'figure'+str(nth_window)+'.png')

def plot_stability(savepath, freq_matrix, alphas, nth_window, auc):
    # Assuming a square matrix
    n_genes = len(freq_matrix)
    fig = plt.figure(figsize=(17, 10))
    graph = 1
    x_max = round(np.max(alphas),3)
    x_min = round(np.min(alphas), 3)
    x_mid = round((np.max(alphas)-np.min(alphas))/2.0,3)
    for ii in range(n_genes):
        for jj in range(n_genes):
            stability = freq_matrix[ii, jj, nth_window, :]
            ax = fig.add_subplot(n_genes, n_genes, graph)
            graph+=1
            ax.plot(alphas, stability)
            #ax.set_xlabel('Alpha values', fontsize=8)
            #ax.set_ylabel('Freq', fontsize=8)
            ax.set_title('AUC: %0.5f' %auc[ii,jj], fontsize=9)
            ax.yaxis.set_ticks([0.0, 0.5, 1.0])
            ax.xaxis.set_ticks([x_min, x_mid, x_max])
            ax.tick_params(axis='both', labelsize=8)
    fig.subplots_adjust(left=0.04, bottom=0.03, right=0.97, top=0.97, hspace=0.8, wspace=0.5)
    plt.savefig(savepath)

