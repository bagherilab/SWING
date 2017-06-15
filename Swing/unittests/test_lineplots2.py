import matplotlib
matplotlib.use('Agg')
from Swing.util.LinePlot import LinePlot
from Swing.util.Analyzer import Analyzer
from numpy import mean
import pdb,random

from Swing.tdSwing import tdSwing
from Swing.util.Evaluator import Evaluator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def main(data_folder, output_path, target_dataset, my_statistic, td_score):
    #Analyzer computes AUROC/AUPR/Cragging Scores and organizes it in a table

    analyzer = Analyzer(data_folder)

    #identify the x axis in analyzer
    time_vec = analyzer.current_roller.time_vec.tolist()

    lp = LinePlot()


    my_df = analyzer.overall_df
    grouped = my_df.groupby(['window_width','window_index'])

    max_series = []
    min_series = []
    mean_series = []
    ## iterate through window_sizes 
    unique_window_sizes = list(set(analyzer.overall_df['window_width'].tolist()))
    lp.set_x_values(unique_window_sizes)
    for color_index, window_size in enumerate(unique_window_sizes):
        series_y = []
        ## get unique indices
        unique_indices = my_df[my_df['window_width']==window_size]['window_index'].unique()
        unique_indices.sort()
        for index in unique_indices:
            value = grouped.get_group((window_size, index)).mean()[my_statistic]
            series_y.append(value)
            
        ## plot horizontal line for the maximum window size
        if window_size == analyzer.current_roller.overall_width:
            lp.plot_horizontal_line(value, 19, "Status Quo")

        else:
            unique_indices = unique_indices.tolist()
            time_values = [time_vec[x] for x in unique_indices]
            max_series.append(max(series_y))
            mean_series.append(mean(series_y))
            min_series.append(min(series_y))

    #print maximum window
    unique_window_sizes.pop()

    lp.plot_window_series(max_series,0, "Max",x_values=unique_window_sizes)
    lp.plot_window_series(mean_series,1, "Mean",x_values=unique_window_sizes)
    lp.plot_window_series(min_series,2, "Min",x_values=unique_window_sizes)


    ## print best cragging score
    cragged_window = analyzer.predict_best_window()
    lp.plot_horizontal_line(cragged_window[my_statistic].values, 3, 'best crag')
    # plot the tdroller score
    lp.plot_horizontal_line(td_score,17, 'tdRoller')
    lp.add_formatting(y_label=my_statistic)        


    lp.save_plot(output_path, '_'+my_statistic)
    #grouped.get_group((2,2)).mean()['aupr']



def get_td_stats(file_path, td_window = 6): 
    gene_start_column = 1
    time_label = "Time"
    separator = "\t"
    gene_end = None

    df = pd.read_csv(file_path,sep=separator)
    current_gold_standard = file_path.replace("timeseries.tsv","goldstandard.tsv")
    node_list = df.columns.tolist()
    node_list.pop(0)

    evaluator = Evaluator(current_gold_standard, '\t', node_list=node_list)
    true_edges = evaluator.gs_flat.tolist()
    pd.options.display.float_format = '{:,.5f}'.format

    #np.random.seed(8)

    tdr = tdSwing(file_path, gene_start_column, gene_end, time_label, separator)
    tdr.zscore_all_data()
    tdr.set_window(td_window)
    tdr.create_windows()
    tdr.augment_windows(min_lag=1, max_lag=4)
    tdr.fit_windows(n_trees=10, show_progress=False)
    tdr.rank_edges(permutation_n=10)
    tdr.compile_roller_edges(self_edges=True)

    tdr.full_edge_list.loc[tdr.full_edge_list.p_value>=0.05, 'Importance'] = 0
    tdr.make_static_edge_dict(true_edges, lag_method='median_median')
    df2 = tdr.make_sort_df(tdr.edge_dict, 'lag')
    print len(df2)
    roc_dict, pr_dict = tdr.score(df2)
    print roc_dict['auroc'][-1]
    print pr_dict['aupr'][-1]#+(1-pr_dict['recall'][-1])
    #tdr.plot_scoring(roc_dict, pr_dict)
    return((roc_dict['auroc'][-1],pr_dict['aupr'][-1]))

data_folder = "/projects/p20519/roller_output/optimizing_window_size/RandomForest/janes"
output_path = "/home/jjw036/Swing/janes"
target_dataset = "/projects/p20519/Swing/data/invitro/janes_timeseries.tsv"
#target_dataset = "/projects/p20519/Swing/data/dream4/ecoli_timeseries.tsv"
my_statistic = 'aupr'
roc,pr = get_td_stats(target_dataset, 5)
pdb.set_trace()
td_score = pr

main(data_folder, output_path, target_dataset, 'auroc', roc)
main(data_folder, output_path, target_dataset, 'aupr', pr)

