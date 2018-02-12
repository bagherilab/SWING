import pandas as pd
from Swing import Swing


if __name__ == "__main__":
    # Load data
    gene_start_column = 1
    time_label = "Time"
    separator = "\t"
    gene_end = None
    file_path = "data/Ecoli_10_node/Ecoli10-1_dream4_timeseries.tsv"
    gold_standard_file = "data/Ecoli_10_node/Ecoli10-1_goldstandard.tsv"
    df = pd.read_csv(file_path, sep=separator)

    # Set SWING parameters
    k_min = 1
    k_max = 3
    w = 10
    method = 'RandomForest'
    trees = 10

    # Initialize a SWING object
    sg = Swing(file_path, gene_start_column, gene_end, time_label, separator, min_lag=k_min,
               max_lag=k_max, window_width=w, window_type=method)

    sg.zscore_all_data()
    sg.create_windows()
    sg.optimize_params()

    sg.fit_windows(n_trees=trees, show_progress=False, n_jobs=-1)
    sg.compile_edges(self_edges=False)

    sg.make_static_edge_dict(self_edges=False, lag_method='mean_mean')
    ranked_edges = sg.make_sort_df(sg.edge_dict)

    roc_dict, pr_dict = sg.score(ranked_edges, gold_standard_file=gold_standard_file)
    print(roc_dict['auroc'][-1])
