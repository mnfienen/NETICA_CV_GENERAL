import xml.etree.ElementTree as ET
import numpy as np

###################
# Tools for reading XML input file
###################

class netica_scenario:
    def __init__(self):
        self.name = None
        self.nodesIn = None
        self.response = None

class input_parameters:
    # class and methods to read and parse XML input file
    def __init__(self,infile):
        self.infile = infile
        try:
            inpardat = ET.parse(self.infile)
        except:
            raise(FileOpenFail(self.infile)) 
        inpars = inpardat.getroot()
        self.baseNET = inpars.findall('.//input_files/baseNET')[0].text
        self.baseCAS = inpars.findall('.//input_files/baseCAS')[0].text
        self.pwdfile = inpars.findall('.//input_files/pwdfile')[0].text
        self.numfolds = int(inpars.findall('.//kfold_data/numfolds')[0].text)
        self.scenario = netica_scenario()
        self.scenario.name =  inpars.findall('.//scenario/name')[0].text
        self.scenario.nodesIn = []
        for cv in inpars.findall('.//scenario/input'):
            self.scenario.nodesIn.append(cv.text)
        self.scenario.response = inpars.findall('.//scenario/response')[0].text



###################
# Tools for k-fold setup
###################
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

#################
# Error classes
#################

# -- cannot open an input file
class FileOpenFail(Exception):
    def __init__(self,filename):
        self.fn = filename
    def __str__(self):
        return('\n\nCould not open %s.' %(self.fn))    