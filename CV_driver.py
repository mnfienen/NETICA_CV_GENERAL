import pythonNeticaUtils as pyn
import CV_tools as CVT
import numpy as np
import pickle
'''
CV_driver.py

a cross-validation driver for Netica
a m!ke@usgs joint
'''


############
# CONFIGURATION FILE NAME
parfile = 'example.xml'
############
cdat = pyn.pynetica()
cdat.probpars = CVT.input_parameters(parfile)

cdat.start_environment(cdat.probpars.pwdfile)
# Initialize a pynetica instance/env using password in a text file

# read in the data from a cas file
cdat.read_cas_file(cdat.probpars.baseCAS)

# determine the number of data points
cdat.numfolds = cdat.probpars.numfolds
# create the folds desired
cdat.allfolds = CVT.all_folds()
cdat.allfolds.k_fold_maker(cdat.N,cdat.numfolds)

# run the predictions using the base net --> 
cdat.predictBayes(cdat.probpars.baseNET)
# write the results to a post-processing world
cdat.PredictBayesPostProc()

# rock the cross-validation work 
if cdat.probpars.CVflag:
    # now run for each fold with both retained and leftout indices
    for cfold in np.arange(cdat.probpars.numfolds):
        for i in ['calibration','validation']:
            print i
# first need to sanitize away any ctypes/Netica pointers
cdat.sanitize()
# now dump into a pickle file
outfilename = parfile[:-4] + '_cdat.pkl'
print 'Dumping cdat to pickle file --> %s' %(outfilename)
ofp = open(outfilename,'wb')
pickle.dump(cdat,ofp)
ofp.close()