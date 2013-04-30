import numpy as np
import os
import pythonNeticaUtils as pyn

'''
CV_driver.py

a cross-validation driver for Netica
a m!ke@usgs joint
'''
# Initialize a pynetica instance/env using password in a text file
cdat = pyn.pynetica('mikeppwd.txt')

cas_dat_root = 