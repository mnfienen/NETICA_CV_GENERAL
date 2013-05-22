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

cdat.rebuild_net(cdat.probpars.baseNET,
                 cdat.probpars.baseCAS,
                 100,
                 'testnet2.neta',
                 True)
# run the predictions using the current net --> this will need to get looped...
cdat.predictBayes('testnet2.neta',True)

# write the results to a post-processing world
cdat.PredictBayesPostProc()