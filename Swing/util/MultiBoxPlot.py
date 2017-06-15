import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, pdb
from BoxPlot import BoxPlot

class MultiBoxPlot(BoxPlot):
    
    def __init__(self):
        BoxPlot.__init__(self)
        self.axes_list = []
        self.axes_list.append(plt.subplot(2,1,1))
        self.axes_list.append(plt.subplot(2,1,2))

    
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
        self.axes.set_title(title, fontsize=25)
        self.axes.set_aspect(25)

        self.axes.set_ylabel(y_label, fontsize=30)

        ylabels = self.axes.get_yticklabels()
        xlabels = self.axes.get_xticklabels()
        for label in (self.axes.get_xticklabels()):
            label.set_fontsize(18)
            label.set_rotation('vertical')
        for label in (self.axes.get_yticklabels()):
            label.set_fontsize(20)
        for l in self.axes.get_xticklines() + self.axes.get_yticklines():
            l.set_markersize(0)

