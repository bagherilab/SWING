import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, pdb
import scipy
from Swing.util.BasePlot import BasePlot

class BoxPlot(BasePlot):
    
    def __init__(self):
        BasePlot.__init__(self)
        self.meanpointprops = dict(marker='D', markersize=6)
        self.flierprops = dict(marker='o', markersize=6, markerfacecolor='black', markeredgecolor='black', linewidth=8.0)
        self.boxprops = dict(color='black', linewidth=3.0)
        self.whiskerprops = dict(color='black', linewidth=2.0)
        self.capprops = self.whiskerprops
        self.medianprops = dict(color='blue', linewidth=2.5)
    
    def plot_box(self, y_values, labels = None):
        """
        Plots the summary data
        :param y_values: list of np vectors
        :param label: int or str
                      example: the window size
        :return fig obj:
        """
        bp=self.axes.boxplot(y_values, labels=labels, widths = 0.3,medianprops = self.medianprops, whiskerprops=self.whiskerprops,flierprops=self.flierprops, meanprops=self.meanpointprops,showmeans=True, boxprops=self.boxprops, capprops=self.capprops)
        return(bp)

    def add_formatting(self, title, y_label):
        self.axes.annotate(title, xy=(0.5, 1.01), xycoords='axes fraction', horizontalalignment='center', fontsize = 25)
        #self.axes.set_aspect(25)

        self.axes.set_ylabel(y_label, fontsize=30)

        #self.axes.set_ylim([-0.4,0.6])
        #self.axes.yaxis.set_ticks(np.arange(-0.4, 0.6, 0.1))
        ylabels = self.axes.get_yticklabels()
        xlabels = self.axes.get_xticklabels()
        for label in (self.axes.get_xticklabels()):
            label.set_fontsize(18)
            label.set_rotation('vertical')
        for label in (self.axes.get_yticklabels()):
            label.set_fontsize(20)
        for l in self.axes.get_xticklines() + self.axes.get_yticklines():
            l.set_markersize(0)

    def add_significance(self, mann_whitney_results, style = 'separate', reset=0.06):        
        counter = 0.01
        for result in mann_whitney_results:
            if counter > 0.05:
                counter = 0.01
            index_x = result[0]
            index_y = result[1]
            significance = result[2]
            y_limits = self.axes.get_ylim()

    
            if style == 'cascade':
                if significance < 0.05:
                    self.axes.hlines(y=counter, xmin=index_x+1, xmax=index_y+1, color = "black")
                    if significance < 0.01:
                        self.axes.annotate('**', xy=((index_x+index_y+2)/2, counter-0.075), xycoords='data', horizontalalignment='center', fontsize = 20, weight='heavy', color = "black")
                    else:
                        self.axes.annotate('*', xy=((index_x+index_y+2)/2, counter-0.075), xycoords='data', horizontalalignment='center', fontsize = 20, weight='heavy', color = "black")
                    counter = counter + 0.01
            elif style == 'separate':
                if significance < 0.05:
                    self.axes.hlines(y=y_limits[1]-0.05, xmin=index_x+1, xmax=index_y+1, color = "black")
                    if significance < 0.01:
                        self.axes.annotate('**', xy=((index_x+index_y+2)/2, y_limits[1]-0.075), xycoords='data', horizontalalignment='center', fontsize = 20, weight='heavy', color = "black")
                    else:
                        self.axes.annotate('*', xy=((index_x+index_y+2)/2, y_limits[1]-0.075), xycoords='data', horizontalalignment='center', fontsize = 20, weight='heavy', color = "black")
                    
        return()
    
    def sigtest(self, data_list, score):
        results = []
        for test in score:
            index_x = test[0]
            index_y = test[1]
            test_result = scipy.stats.mannwhitneyu(data_list[index_x], data_list[index_y])
            p_value = test_result[1]*2
            results.append( (index_x, index_y, p_value) )
        return(results)
          
    def add_sections(self, box_plots_per_section, annotation_per_section, offset=0.05):
        x_lim = self.axes.get_xlim()
        total_boxplots = x_lim[1] - 0.5
        line_coords = [x for x in range(0,int(total_boxplots),box_plots_per_section)]
        #pop out the first one
        #line_coords = line_coords[1:]
        annotation_location = list(np.linspace(0, 1, total_boxplots/box_plots_per_section, endpoint=False))
        line_annotation = zip(line_coords, annotation_per_section, annotation_location)

        for line, text, loc in line_annotation:
            self.axes.axvline(x=line+0.5, color = "gray")
            self.axes.annotate(text, xy=(loc+offset, .95), xycoords='axes fraction', horizontalalignment='center', fontsize = 20, weight='heavy', color = "gray")
        return(True)

