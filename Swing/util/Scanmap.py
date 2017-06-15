class Scanmap:
"""A heatmap and line plot combined into one figure"""

    def __init__(self, dim = None):
        if dim:
            self.set_dimensions(dim)
        else:
            default_dim = { 'gp_left': 0.2,
                            'gp_bottom': 0.1,
                            'gp_width': 0.7,
                            'gp_height': 0.2,
                            'padding': 0.01,
                            'numTFs': 20,
                            'dm_left': 0.2,
                            'dm_bottom': 0.32,
                            'dm_width':0.7,
                            'box_height':0.03,
                            'dm_height':0.6 }
            self.set_dimensions(default_dim)

        
        #initialize colormap
        self.tableau20 = [((152,223,138),(31, 119, 180), (174, 199, 232), (255, 127, 14),(255, 187, 120),  (44, 160, 44), (255, 152, 150),(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(214,39,40)]
        
        for i in range(len(tableau20)):
            r,g,b = self.tableau20[i]
            self.tableau20[i] = (r/255., g/255., b/255.)

        #initialize axes
        f = plt.figure(figsize=(10,10))
        d = self.dimensions
        axarr2 = f.add_axes(d['gp_left'],d['gp_bottom'],d['gp_width'],d['gp_height'])
        axarr1 = f.add_axes(d['dm_left'],d['dm_bottom'],d['dm_width'],d['dm_height'])

        
    
    def set_dimensions(self, dim_dict):
        self.dimensions = dim_dict
        return(dim_dict)

        
