import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
#--modify default matplotlib settings
mpl.rcParams['font.sans-serif']          = 'Univers 57 Condensed'
mpl.rcParams['font.serif']               = 'Times'
mpl.rcParams['font.cursive']             = 'Zapf Chancery'
mpl.rcParams['font.fantasy']             = 'Comic Sans MS'
mpl.rcParams['font.monospace']           = 'Courier New'
mpl.rcParams['mathtext.default']         = 'regular'
mpl.rcParams['pdf.compression']          = 0
mpl.rcParams['pdf.fonttype']             = 42
#--figure text sizes
mpl.rcParams['legend.fontsize']  = 18
mpl.rcParams['axes.labelsize']   = 18
mpl.rcParams['xtick.labelsize']  = 18
mpl.rcParams['ytick.labelsize']  = 18

# ############################
# USER DATA
probroot = 'glass_sqrt_dist'
allstats = ['min','max','mean','median']
allmetrics = ['skillMean',
        'rmseMean',
        'meanErrMean',
        'meanAbsErrMean',
        'skillML',
        'rmseML',
        'meanErrML',
        'meanAbsErrML',
        'LogLoss',
        'ErrorRate',
        'QuadraticLoss']
numsets = 10
numfolds = 10
# ############################
figdir = probroot + '_plots'
if os.path.exists(os.path.join(os.getcwd(),figdir)):
    shutil.rmtree(os.path.join(os.getcwd(),figdir))
os.mkdir(figdir)

def plotdat(allmetrics):
    plotdat = dict()
    for cmet in allmetrics:
        plotdat[cmet] = np.empty(0)
    return plotdat

class alldat:
    def __init__(self,calval,filenames,numsets,allstats,allmetrics):
        self.calval = calval
        self.allstats = allstats
        self.numsets = numsets
        self.allmetrics = allmetrics
        self.infiles = filenames
        self.setnum = np.arange(numsets)
        self.outdata = dict()
        self.indat = list()
        
    def readinfile(self,filenum):
        self.indat.append(np.genfromtxt(self.infiles[filenum],dtype=None,names=True,skiprows=4))
        if filenum == 0:
            self.allresponses = np.unique(self.indat[0]['Response'])
    
    def setup_plotdata(self):
        for cstat in self.allstats:
            self.outdata[cstat] = dict()
            for i in self.allresponses:
                self.outdata[cstat][i] = plotdat(self.allmetrics)
                                   
    
    def populate_plotdata(self):
        for cstat in self.allstats:
            for cset in np.arange(self.numsets):
                for cmet in self.allmetrics:
                    for cres in self.allresponses:
                        crow = np.intersect1d(np.where(self.indat[cset]['Stat']==cstat)[0], 
                                    np.where(self.indat[cset]['Response']==cres)[0])
                        self.outdata[cstat][cres][cmet] = np.hstack((self.outdata[cstat][cres][cmet],
                                                                     self.indat[cset][cmet][crow]))
                    
def make_plots(CALdat,VALdat,figdir):
    for cstat in CALdat.allstats:
        for cmet in CALdat.allmetrics:
            for cres in CALdat.allresponses:
                print 'plotting --> %s_%s_%s.pdf' %(cstat,cmet,cres)
                outfig = plt.figure()
                plt.hold(True)
                plt.plot(np.arange(CALdat.numsets)+1,CALdat.outdata[cstat][cres][cmet],'r-x')
                plt.plot(np.arange(VALdat.numsets)+1,VALdat.outdata[cstat][cres][cmet],'b-x')
                plt.title('%s of %s for %s over %d sets' %(cstat,cres,cmet,CALdat.numsets))
                plt.xlabel('Sets')
                plt.ylabel(cmet)
                plt.grid(True)
                plt.legend(['Calibration','Validation'],loc='best')
                if 'skill' in cmet:
                    plt.ylim((0.0,1.0))
                plt.savefig(os.path.join(figdir,'%s_%s_%s.pdf' %(cstat,cmet,cres)))


CALroots = list()
VALroots = list()
for i in np.arange(numsets):
    CALroots.append('%s_set%d_kfold_stats_CAL_%d_folds_SUMMARY.dat' %(probroot,i+1,numfolds))
    VALroots.append('%s_set%d_kfold_stats_VAL_%d_folds_SUMMARY.dat' %(probroot,i+1,numfolds))

# set up preliminary structure to read in all the data
alldata = dict()
alldata['CAL'] = alldat('CAL',CALroots,numsets,allstats,allmetrics)
alldata['VAL'] = alldat('VAL',VALroots,numsets,allstats,allmetrics)

# read in the data
for i in np.arange(numsets):
    alldata['CAL'].readinfile(i)
    alldata['VAL'].readinfile(i)

# based on what was read in, set up the storage for plotting data
alldata['CAL'].setup_plotdata()
alldata['VAL'].setup_plotdata()

# now populate the data fields
alldata['CAL'].populate_plotdata()
alldata['VAL'].populate_plotdata()

# finally make the plots
make_plots(alldata['CAL'],alldata['VAL'],figdir)