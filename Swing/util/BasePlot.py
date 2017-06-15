import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BasePlot:

    def __init__(self):
        self.f = plt.figure(figsize=(10,10))
        self.axes = self.f.gca()

        #initialize colormap
        self.tableau20 = [(152,223,138),(31, 119, 180), (174, 199, 232), (255, 127, 14),(255, 187, 120),  (44, 160, 44), (255, 152, 150),(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(214,39,40)]
        
        for i in range(len(self.tableau20)):
            r,g,b = self.tableau20[i]
            self.tableau20[i] = (r/255., g/255., b/255.)    

    def save_plot(self, folder, tag):
        """
        Saves the plot in designated area.
        :param folder: string
        :param tag: string
        :return self.f: returns the figure
        """
        image_save = folder + tag + ".ps"
        self.f = self.f.savefig(image_save,format = "ps", bbox_inches='tight')
        return(self.f)

    def add_formatting(self):
        pass

