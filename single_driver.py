import pythonNeticaUtils as pyn
import CV_tools as CVT
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

cdat.rebuild_net(cdat.probpars.baseNET,
                 cdat.probpars.baseCAS,
                 100,
                 'testnet2.neta',
                 True)
# run the predictions using the current net --> this will need to get looped...
cdat.predictBayes('testnet2.neta')

# write the results to a post-processing world
cdat.PredictBayesPostProc()

# dump results into a pickle file for later plotting
# N.B. --> this file assume the same root as the parfile <example>.xml

# first need to sanitize away any ctypes/Netica pointers
cdat.sanitize()
# now dump into a pickle file
outfilename = parfile[:-4] + '_cdat.pkl'
print 'Dumping cdat to pickle file --> %s' %(outfilename)
ofp = open(outfilename,'wb')
pickle.dump(cdat,ofp)
ofp.close()