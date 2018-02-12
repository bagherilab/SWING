SWING: Sliding Window Inference for Network Generation
=======================================

SWING is a network inference framework that identifies associations between genes using time-series gene expression data. SWING is based on multivariate Granger causality and sliding window regression and is currently implemented in python. 

Documentation
-------------
For source code-related documentation, check our sphinx documentation in the docs/\_build/\html folder. For general implementation details, please read the Supporting Information section of Finkle et al 2018.

Examples
--------
Please see the jupyter notebook in the examples/ for a working pipeline.

Citing
------

Dependencies
------------

- Python 3.6+

### Required

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [pandas](http://pandas.pydata.org/)

- [scikit-learn](http://scikit-learn.org/stable/)

Installation
------------
SWING is available via github and is a python package. Here is an example installation script:
git clone git@github.com:bagherilab/SWING.git
cd SWING
python setup.py
 
Development
-----------
Please report bugs or submit your suggestions on the official SWING git repo:
https://github.com/bagherilab/SWING

Authors
-----------

- [Justin Finkle](https://github.com/justinfinkle)

- [Jia Wu](https://github.com/jiawu)

- [Dr. Neda Bagheri](http://www.mccormick.northwestern.edu/research-faculty/directory/profiles/bagheri-neda.html)
