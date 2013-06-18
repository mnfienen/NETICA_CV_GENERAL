import numpy as np
import scipy.stats.mstats as scpstat

class netica_binning:
    def __init__(self,x,n):
        self.x = x #vector of values to bin
        self.n = n # number of bins

    def bin_thresholds(self):
        a = np.min( self.x )
        b = np.max( self.x )
        self.probs = np.linspace(0,100.0,self.n+1)/100.
        self.binlevels = scpstat.mquantiles(self.x,self.probs)

