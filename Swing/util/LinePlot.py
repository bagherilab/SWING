import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, pdb
from BasePlot import BasePlot

class LinePlot(BasePlot):
    
    def set_x_values(self, x_values):
        """
        Sets the x values (ie time points)
        :param x_values: list of str/ints
        """
        self.x_values = map(int,x_values)

    
    def plot_window_series(self, y_values, color_index, label, x_values = None):
        """
        Plots points that are interpolated with a line.
        :param y_values: list
        :param label: int or str
                      example: the window size
        """
        #if not x_values:
            #x_values = self.x_values
        if color_index > 19:
            color_index = color_index%20
        self.axes.plot(x_values,y_values, linestyle='-', color = self.tableau20[color_index], label = str(label), linewidth=5)

    def plot_vertical_line(self, x_value,color_index, label):
        """
        Plots a vertical line.
        :param x_value: int
        :param label: int or str
                      example: window size
        :param color_index: corresponding color index on tableau20
        """
        if color_index > 19:
            color_index = color_index%20
        self.axes.axvline(x=x_value, linestyle='--',color=self.tableau20[color_index],label="WS "+str(label), linewidth=3)

    
    def plot_horizontal_line(self, y_value, color_index, label):
        """
        Plots a horizontal line.
        :param y_value: int
        :param label: int or str
                      example: the window size
        """
        my_x = [min(self.x_values), max(self.x_values)]
        my_y = [y_value,y_value]
        if color_index > 19:
            color_index = color_index%20
        self.axes.plot(my_x, my_y, linestyle='--',color=self.tableau20[color_index],label="WS "+str(label), linewidth=3)


    def add_formatting(self, min_tick=0,max_tick=1200, interval=200):
        #legend
        box = self.axes.get_position()
        self.axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        self.axes.legend(fontsize=8,bbox_to_anchor=(0.5, -0.2), loc='upper center',ncol=7,fancybox=True, shadow=True)
        #labels
        self.axes.set_ylabel('AUROC')
        self.axes.set_xlabel('Time (min)')
        xlabels = self.axes.get_xticklabels()
        ylabels = self.axes.get_yticklabels()
        for label in xlabels:
            label.set_rotation(90)
            label.set_fontsize(16)
        for label in (self.axes.get_yticklabels()):
            label.set_fontsize(16)
        for l in self.axes.get_xticklines() + self.axes.get_yticklines():
            l.set_markersize(0)

        line_ticks = np.arange(min_tick,max_tick,interval)
        self.axes.xaxis.set_ticks(line_ticks)
 
    def add_formatting_auroc(self, min_tick=0,max_tick=1200, interval=200):
        #legend
        box = self.axes.get_position()
        self.axes.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
        self.axes.legend(fontsize=8,bbox_to_anchor=(0.5, -0.2), loc='upper center',ncol=7,fancybox=True, shadow=True)
        #labels
        self.axes.set_ylabel('True Positive Rate', fontsize = 16)
        self.axes.set_xlabel('False Positive Rate', fontsize = 16)
        xlabels = self.axes.get_xticklabels()
        ylabels = self.axes.get_yticklabels()
        for label in xlabels:
            label.set_rotation(90)
            label.set_fontsize(16)
        for label in (self.axes.get_yticklabels()):
            label.set_fontsize(16)
        for l in self.axes.get_xticklines() + self.axes.get_yticklines():
            l.set_markersize(0)

        line_ticks = np.arange(min_tick,max_tick,interval)

