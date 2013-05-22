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
cdat.pred,cdat.NETNODES = cdat.predictBayes(cdat.probpars.baseNET,cdat.casdata,cdat.N)
# get the stats from the predictions
print 'getting stats'
cdat.pred = cdat.PDF2Stats(cdat.pred,cdat.probpars.scenario.response,False,-999,alpha=0.1)
# write the results to a post-processing world
cdat.PredictBayesPostProc()



# rock the cross-validation work 
if cdat.probpars.CVflag:
    print '\n' * 2 + '#'*20 +'\n Performing k-fold cross-validation'
    # make the necessary case files
    print '\nMaking the casefiles for all folds'
    cdat.cross_val_make_cas_files()
    # now build all the nets
    for cfold in np.arange(cdat.probpars.numfolds):
        # rebuild the net
        cname = cdat.allfolds.casfiles[cfold]
        cdat.rebuild_net(cdat.probpars.baseNET,
                         cname,
                         cdat.probpars.voodooPar,
                         cname[:-4] + '.neta',
                         cdat.probpars.EMflag)
        
'''    
        for i in ['calibration','validation']:
            print i
'''            
            
            
            
# first need to sanitize away any ctypes/Netica pointers
cdat.sanitize()
# now dump into a pickle file
outfilename = parfile[:-4] + '_cdat.pkl'
print 'Dumping cdat to pickle file --> %s' %(outfilename)
ofp = open(outfilename,'wb')
pickle.dump(cdat,ofp)
ofp.close()