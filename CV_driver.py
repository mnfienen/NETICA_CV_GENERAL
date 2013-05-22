import pythonNeticaUtils as pyn
import CV_tools as CVT
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

# run the predictions using the current net --> this will need to get looped...
cdat.predictBayes(cdat.probpars.baseNET,True)

# write the results to a post-processing world
cdat.PredictBayesPostProc()