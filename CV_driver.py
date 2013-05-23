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
# initialize
cdat = pyn.pynetica()
# read in the problem parameters
cdat.probpars = CVT.input_parameters(parfile)

# Initialize a pynetica instance/env using password in a text file
cdat.start_environment(cdat.probpars.pwdfile)

# read in the data from a base cas file
cdat.read_cas_file(cdat.probpars.baseCAS)

# determine the number of data points
cdat.numfolds = cdat.probpars.numfolds
# create the folds desired
cdat.allfolds = CVT.all_folds()
cdat.allfolds.k_fold_maker(cdat.N,cdat.numfolds)

# run the predictions using the base net --> 
cdat.basepred,cdat.NETNODES = cdat.predictBayes(cdat.probpars.baseNET,cdat.N,cdat.casdata)
# write the results to a post-processing world
cdat.PredictBayesPostProc(cdat.basepred,
                          cdat.probpars.scenario.name + '_base_stats.dat',
                          cdat.probpars.baseCAS)

# if requested, perform K-fold cross validation
if cdat.probpars.CVflag:
    print '\n' * 2 + '#'*20 +'\n Performing k-fold cross-validation'
    # set up for cross validation
    print '\nSetting up cas files and file pointers for cross validation'
    kfoldOFP_Val,kfoldOFP_Cal = cdat.cross_val_setup()
    # now build all the nets
    for cfold in np.arange(cdat.probpars.numfolds):
        # rebuild the net
        cname = cdat.allfolds.casfiles[cfold]
        cdat.rebuild_net(cdat.probpars.baseNET,
                         cname,
                         cdat.probpars.voodooPar,
                         cname[:-4] + '.neta',
                         cdat.probpars.EMflag)
        # make predictions for both validation and calibration data sets
        print '*'*5 + 'Calibration predictions' + '*'*5
        cdat.allfolds.calpred[cfold],cdat.allfolds.calNODES[cfold] = (
            cdat.predictBayes(cname[:-4] + '.neta',
                              cdat.allfolds.calN[cfold],
                              cdat.allfolds.caldata[cfold]))
        print '*'*5 + 'End Calibration predictions' + '*'*5  + '\n\n'      

        print '*'*5 + 'Start Validation predictions' + '*'*5        
        cdat.allfolds.valpred[cfold],cdat.allfolds.valNODES[cfold] = (
            cdat.predictBayes(cname[:-4] + '.neta',
                              cdat.allfolds.valN[cfold],
                              cdat.allfolds.valdata[cfold]))
        print '*'*5 + 'End Validation predictions' + '*'*5   + '\n\n'      

        cdat.PredictBayesPostProcCV(cdat.allfolds.valpred[cfold],cfold,kfoldOFP_Val,'Validation')
        cdat.PredictBayesPostProcCV(cdat.allfolds.calpred[cfold],cfold,kfoldOFP_Cal,'Calibration')

    kfoldOFP_Cal.close()
    kfoldOFP_Val.close()
    
# first need to sanitize away any ctypes/Netica pointers
cdat.sanitize()
# now dump into a pickle file
outfilename = parfile[:-4] + '_cdat.pkl'
print 'Dumping cdat to pickle file --> %s' %(outfilename)
ofp = open(outfilename,'wb')
pickle.dump(cdat,ofp)
ofp.close()