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
allsets = ['2','3','4','5','6','7','8','4_5','4_6','4_8','4_10','5_6','5_8','5_10']
probroot = 'glacialbins'
allstats = ['min','max','mean','median']
allmetrics = ['skillMean',
        'rmseMean',
        'meanErrMean',
        'meanAbsErrMean',
        'skillML',
        'rmseML',
        'meanErrML',
        'meanAbsErrML']
numsets = 8
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
    def __init__(self,calval,filenames,allsets,allstats,allmetrics):
        self.calval = calval
        self.allstats = allstats
        self.allsets = allsets
        self.allmetrics = allmetrics
        self.infiles = filenames
        self.setnum = np.arange(1,numsets)
        self.outdata = dict()
        self.indat = dict()
        
    def readinfile(self,cset):
        self.indat[cset] = np.genfromtxt(self.infiles[cset],dtype=None,names=True,skiprows=4)
        if cset == self.allsets[0]:
            self.allresponses = np.unique(self.indat[cset]['Response'])
    
    def setup_plotdata(self):
        for cstat in self.allstats:
            self.outdata[cstat] = dict()
            for i in self.allresponses:
                self.outdata[cstat][i] = plotdat(self.allmetrics)
                                   
    
    def populate_plotdata(self):
        for cstat in self.allstats:
            for cset in self.allsets:
                for cmet in self.allmetrics:
                    for cres in self.allresponses:
                        crow = np.intersect1d(np.where(self.indat[cset]['Stat']==cstat)[0], 
                                    np.where(self.indat[cset]['Response']==cres)[0])
                        self.outdata[cstat][cres][cmet] = np.hstack((self.outdata[cstat][cres][cmet],
                                                                     self.indat[cset][cmet][crow]))
                    
def make_plots(CALdat,VALdat,figdir):
    for cstat in CALdat.allstats:
        print cstat
        for cmet in CALdat.allmetrics:
            print cmet
            for cres in CALdat.allresponses:
                print cres
                print 'plotting --> %s_%s_%s.pdf' %(cstat,cmet,cres)
                outfig = plt.figure()
                ax = outfig.add_subplot(111)
                plt.hold(True)
                plt.plot(CALdat.outdata[cstat][cres][cmet],'r-x')
                plt.plot(VALdat.outdata[cstat][cres][cmet],'b-x')
                plt.title('%s of %s for %s over bins' %(cstat,cres,cmet))
                plt.xlabel('Sets')
                plt.xticks(np.arange(len(CALdat.allsets)),CALdat.allsets,rotation=45)
                plt.yticks(np.linspace(0,1,11))
                plt.ylabel(cmet)
                plt.grid(True)
                plt.legend(['Calibration','Validation'],loc='best')
                if 'skill' in cmet:
                    plt.ylim((0.0,1.0))
                plt.savefig(os.path.join(figdir,'%s_%s_%s.pdf' %(cstat,cmet,cres)))


CALroots = dict()
VALroots = dict()
for cset in allsets:
    CALroots[cset] = '%s%s_kfold_stats_CAL_%d_folds_SUMMARY.dat' %(probroot,cset,numfolds)
    VALroots[cset] = '%s%s_kfold_stats_VAL_%d_folds_SUMMARY.dat' %(probroot,cset,numfolds)

# set up preliminary structure to read in all the data
alldata = dict()

alldata['CAL'] = alldat('CAL',CALroots,allsets,allstats,allmetrics)
alldata['VAL'] = alldat('VAL',VALroots,allsets,allstats,allmetrics)

# read in the data
for cset in allsets:
    alldata['CAL'].readinfile(cset)
    alldata['VAL'].readinfile(cset)

# based on what was read in, set up the storage for plotting data
alldata['CAL'].setup_plotdata()
alldata['VAL'].setup_plotdata()

# now populate the data fields
alldata['CAL'].populate_plotdata()
alldata['VAL'].populate_plotdata()

# finally make the plots
make_plots(alldata['CAL'],alldata['VAL'],figdir)