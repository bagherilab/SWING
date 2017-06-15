import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import unittest
import numpy as np
import Swing
import pdb
import Swing.util.Grapher as Grapher
import pandas as pd
from Swing.util.Evaluator import Evaluator
from scipy.stats.stats import pearsonr
class TestRFGrapher(unittest.TestCase):
    def setUp(self):
        file_path = "../../data/dream4/insilico_size10_3_timeseries.tsv"
        gene_start_column = 1
        time_label = "Time"
        separator = "\t"
        gene_end = None

        self.roller = Swing.Swing(file_path, gene_start_column, gene_end, time_label, separator,window_type = "RandomForest")
        roller_result_path = "sample_RF_roller_results2.pickle"
        self.roller_results = pd.read_pickle(roller_result_path)
    
    def test_aggr_stats_correlation(self):
        dm_left = 0.2
        dm_bottom = 0.1+0.2+2*0.01
        dm_width = 0.7
        dm_height = 0.03* 10
        ## get AUROC for each window
        roller = self.roller_results
        window_start_list=[]
        auroc_list = []
        rate_list = []
        average_list = []

        for window in roller.window_list:
            window_size = roller.window_width
            min_time = min(window.raw_data['Time'].unique())
            window_start_list.append(min_time)
            gold_standard = roller.file_path.replace("timeseries.tsv","goldstandard.tsv")
            evaluator = Evaluator("../../"+gold_standard,sep="\t")
            sorted_edge_list = window.results_table
            sorted_edge_list.sort(['importance'], ascending=[False], inplace=True)
            sorted_edge_list = sorted_edge_list[np.isfinite(sorted_edge_list['p_value'])]
            tpr,fpr, auroc = evaluator.calc_roc(sorted_edge_list)
            auroc_list.append(auroc[-1])

            #posthoc analysis of rates and averages

            rate_dict = window.get_rate_analysis(1)
            rate_list.append(rate_dict)

            average_list.append(window.get_average())

        ## get the statistics for each gene
        row_labels = window.genes
        row_labels = np.append(row_labels, 'Aggr')
        heatmap_values = pd.DataFrame()

        col_labels = ['Rate (mean)', 'Rate (median)', 'Rate (min)', 'Rate (max)', 'Rate (std)']

        ## dataframe labels: mean values, 
        for index, gene in enumerate(window.genes):
            #correlate auroc with gene mean
            #auroc of a window is in auroc_list
            #auroc_list should have the same length as the rate_dict_list

            #get the mean for gene 1
            item_mean = [rate_dict['mean'][index] for rate_dict in rate_list]
            item_min = [rate_dict['min'][index] for rate_dict in rate_list]
            item_max = [rate_dict['max'][index] for rate_dict in rate_list]
            item_median = [rate_dict['median'][index] for rate_dict in rate_list]
            item_std = [rate_dict['std'][index] for rate_dict in rate_list]
            
            mean_c = pearsonr(item_mean, auroc_list)[0]
            median_c = pearsonr(item_median, auroc_list)[0]
            min_c = pearsonr(item_min, auroc_list)[0]
            max_c = pearsonr(item_max, auroc_list)[0]
            std_c = pearsonr(item_std, auroc_list)[0]
            
            vector = [mean_c, median_c, min_c, max_c, std_c]
            heatmap_values[gene] = vector
        all_means = [rate['all_rates'].mean() for rate in rate_list] 
        all_medians = [np.median(rate['all_rates']) for rate in rate_list] 
        all_mins = [rate['all_rates'].min() for rate in rate_list] 
        all_maxs = [rate['all_rates'].max() for rate in rate_list] 
        all_std = [rate['all_rates'].std(ddof=1) for rate in rate_list]
        all_means_c = pearsonr(all_means, auroc_list)[0]
        all_medians_c = pearsonr(all_medians, auroc_list)[0]
        all_mins_c = pearsonr(all_mins, auroc_list)[0]
        all_maxs_c = pearsonr(all_maxs, auroc_list)[0]
        all_std_c = pearsonr(all_std, auroc_list)[0]

        all_vector = [all_means_c, all_medians_c, all_mins_c, all_maxs_c,all_std_c]
        #rates_list = window.get_rates(1)
        
        heatmap_values['all'] = all_vector
        
        f = plt.figure(figsize=(10,10))
        axarr1 = f.add_axes([dm_left, dm_bottom, dm_width, dm_height])
        my_axis = axarr1.matshow(heatmap_values,cmap=matplotlib.cm.RdBu,aspect='auto')
        my_axi = my_axis.get_axes()
        axarr1.set_xlabel('Genes')
        axarr1.xaxis.set_label_coords(0.5,1.15)
        axarr1.set_yticks(np.arange(heatmap_values.shape[0]))
        axarr1.yaxis.set_ticks_position('left')
        axarr1.set_yticklabels(col_labels)
        axarr1.set_xticks(np.arange(heatmap_values.shape[1]))
        axarr1.xaxis.set_ticks_position('top')
        xlabelsL = axarr1.set_xticklabels(row_labels)
        for label in xlabelsL:
            label.set_rotation(90)
            label.set_fontsize(8)
        for label in (axarr1.get_yticklabels()):
            label.set_fontsize(8)
        for l in axarr1.get_xticklines() + axarr1.get_yticklines():
            l.set_markersize(0)

        image_save = "test_heat_map.png"
        f.savefig(image_save,format="png")
    
    def test_heatmap(self):
        self.roller.set_window(20)
        self.roller.create_windows()
        raw_data=self.roller.raw_data
        figure,ax1 = Grapher.generate_heatmap_from_df(raw_data)


    def test_combo_graphs(self):
        roller = self.roller_results
        raw_data = roller.raw_data
        window_size = roller.window_width
        window_start_list = []
        auroc_list = []
        gp_left = 0.2
        gp_bottom = 0.1
        gp_width = 0.7
        gp_height = 0.2
        
        padding = 0.01
        numTFs = 20
        dm_left = gp_left
        dm_bottom = gp_bottom+gp_height+3*padding
        dm_width = gp_width
        box_height =0.03
        dm_height = box_height*numTFs


        for window in roller.window_list:
            window_size = roller.window_width
            min_time = min(window.raw_data['Time'].unique())
            window_start_list.append(min_time)
            gold_standard = roller.file_path.replace("timeseries.tsv","goldstandard.tsv")
            evaluator = Evaluator("../../"+gold_standard,sep="\t")
            sorted_edge_list = window.results_table
            sorted_edge_list.sort(['p_value'], ascending=[True], inplace=True)
            sorted_edge_list = sorted_edge_list[np.isfinite(sorted_edge_list['p_value'])]
            tpr,fpr, auroc = evaluator.calc_roc(sorted_edge_list)
            auroc_list.append(auroc[-1])
            
        f = plt.figure(figsize=(10,10))
        axarr2 = f.add_axes([gp_left, gp_bottom, gp_width, gp_height])
        axarr1 = f.add_axes([dm_left, dm_bottom, dm_width, dm_height])

        #figure, heatmap = Grapher.generate_heatmap_from_df(self.roller_results.raw_data)
        #axarr1 = figure
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
        my_axis =axarr1.matshow(heatmap_values,cmap=matplotlib.cm.RdBu,aspect='auto')
        my_axi = my_axis.get_axes()
        axarr1.set_yticks(np.arange(heatmap_values.shape[0]))
        axarr1.yaxis.set_ticks_position('left')
        axarr1.set_yticklabels(row_labels)
        axarr1.set_xticks(np.arange(heatmap_values.shape[1]))
        axarr1.xaxis.set_ticks_position('top')
        xlabelsL = axarr1.set_xticklabels(col_labels)
        for label in xlabelsL:
            label.set_rotation(90)
            label.set_fontsize(8)
        for label in (axarr1.get_yticklabels()):
            label.set_fontsize(8)
        for l in axarr1.get_xticklines() + axarr1.get_yticklines():
            l.set_markersize(0)
                
        axarr2.plot(map(int,window_start_list), auroc_list, 'bo', linestyle='-')
        axarr2.plot(map(int,window_start_list), [x+0.3 for x in auroc_list],
            'o', linestyle='-', color = 'r')
        axarr2.xaxis.set_ticks_position('bottom')
        axarr2.xaxis.set_ticks(np.arange(0,1050,50))
        xlabels = axarr2.get_xticklabels()
        ylabels = axarr2.get_yticklabels()
        for label in xlabels:
            label.set_rotation(90)
            label.set_fontsize(8)
        for label in (axarr2.get_yticklabels()):
            label.set_fontsize(8)
        for l in axarr2.get_xticklines() + axarr2.get_yticklines():
            l.set_markersize(0)
        #f.subplots_adjust(wspace=0.001,hspace=0.001,top=None,bottom=None)
        f.savefig("/home/jjw036/Swing/yeast_heatmap2.png")
if __name__ == '__main__':
    unittest.main()
