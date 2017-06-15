"""
Reporter generates a report. Sizes the figures. Adds them to different pages.

methods such as:

  -add_auroc_curve
  -add_aupr_curve
  -add_scanning_plot




"""

class Reporter:
    """Generates a pdf report"""

    def __init__(self):
        self.set_heatmaps(
