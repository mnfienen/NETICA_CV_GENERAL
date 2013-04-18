import numpy as np
import pdb

class all_folds:
    # a class containing leftout and retained indices for cross validation
    
    def __init__(self):
        self.leftout = list()
        self.retained = list()

    
    def k_fold_maker(self,n,k):
    # k_fold index maker
    # a m!ke@usgs joint
    # mnfienen@usgs.gov
    # k_fold_maker(n,k,allfolds)
    # input:
    #   n is the length of the sequence of indices
    #   k is the number of folds to split it into
    #   allfolds is an all_folds class
    # returns an all_folds with each member having k elements
    # allfolds.leftout[i] is the left out indices of fold i
    # allfolds.retained[i] is the kept indices of fold i
        currinds = np.arange(n)
        inds_per_fold = np.int(np.floor(n/k))
        dingleberry = np.remainder(n,k)
        for i in np.arange(k-1):
            allinds = currinds.copy()
            np.random.shuffle(currinds)
            self.leftout.append(currinds[0:inds_per_fold].copy())
            currinds =  np.setdiff1d(allinds,self.leftout[i])
            self.retained.append(currinds.copy())
        
        self.leftout.append(currinds[0:inds_per_fold+dingleberry].copy())
        self.retained.append(np.setdiff1d(np.arange(n),self.leftout[i]))

folds = all_folds()
folds.k_fold_maker(100,3)