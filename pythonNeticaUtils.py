import numpy as np
import os
import re
import ctypes as ct
import platform
import pythonNeticaConstants as pnC
import cthelper as cth
import stats_functions as statfuns
import matplotlib.pyplot as plt
from scipy.stats import nanmean

class nodestruct:
    def __init__(self):
        self.name = None
        self.title = None
        self.beliefs = None
        self.Nbeliefs = None
        self.Likelihood = None
        self.continuous = False
        self.state = []
class parent_inds:
    def __init__(self):
        self.parent_names = list()
        self.parent_indices = list()
        
class experience:
    def __init__(self):
        self.parent_names = list()
        self.parent_states = None
        self.node_experience = list()
        self.case_experience = list()

class netica_test:
    def __init__(self):
        self.logloss = None
        self.errrate = None
        self.quadloss = None
        self.confusion_matrix = None
        self.experience = None
        
class pred_stats:
    def __init__(self):
        self.alpha = None
        self.palpha = None
        self.mean = None
        self.mostProb = None
        self.std = None
        self.median = None
        self.p025 = None
        self.p05 = None
        self.p25 = None
        self.p75 = None
        self.p95 = None
        self.p975 = None
        self.palphaPlus = None
        self.pAlphaMinus = None
        self.meanabserrM = None
        self.meanabserrML = None        
        self.rmseM = None
        self.meaneM = None
        self.rmseML = None
        self.meaneML = None
        self.parent_states = None


class predictions:
    def __init__(self):
        self.z = None
        self.pdf = None
        self.pdfIn = None
        self.ranges = None
        self.rangesplt = None
        self.priorPDF = None
        self.probModelPrior = None
        self.probModelUpdate = None
        self.dataPDF = None
        self.ofp = None
        # statistics go here
        self.stats = None

class statestruct:
    def __init__(self):
        self.obj = None
        self.name = None
        self.numeric = None

class pynetica:
    def __init__(self):
        self.casdata = None
        self.n = None
        self.mesg = ct.create_string_buffer('\000' * 1024)
        self.basepred = None
        self.parent_inds = None
        self.NeticaTests = dict()
        self.NeticaTests['CAL'] = list()
        self.NeticaTests['VAL'] = list()
        
    def sanitize(self):
        print 'Sanitizing pynetica object to remove pointers'
        # code to strip out all ctypes information from SELF to 
        # allow for pickling
        self.n = None
        self.mesg = None
        self.env = None
        
    def start_environment(self,licfile):
        # read in the license file information
        self.licensefile = licfile
        if os.path.exists(self.licensefile):
            self.license = open(self.licensefile,'r').readlines()[0].strip().split()[0]
        else:
            print ("Warning: License File [%s] not found.\n" %(self.licensefile) + 
                   "Opening Netica without licence, which will limit size of nets that can be used.\n" +
                   "Window may become unresponsive.")
            self.license = None         
        self.NewNeticaEnviron()
            
    #############################################
    # Major validation and prediction functions #
    #############################################
    def rebuild_net(self,NetName,newCaseFile,voodooPar,outfilename,EMflag=False):
        '''
         rebuild_net(NetName,newCaseFilename,voodooPar,outfilename)
         a m!ke@usgs joint <mnfienen@usgs.gov>
         function to build the CPT tables for a new CAS file on an existing NET
         (be existing, meaning that the nodes, edges, and bins are dialed)
         INPUT:
               NetName --> a filename, including '.neta' extension
               newCaseFilename --> new case file including '.cas' extension
               voodooPar --> the voodoo tuning parameter for building CPTs
               outfilename --> netica file for newly build net (including '.neta')
               EMflag --> if True, use EM to learn from casefile, else (default)
                         incorporate the CPT table directly
         '''   
        # create a Netica environment
        print 'Rebuilding net: %s using Casefile: %s' %(NetName,newCaseFile)
        # meke a streamer to the Net file
        net_streamer = self.NewFileStreamer(NetName)
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)  
        self.CompileNet(cnet)      
        #get the nodes and their number
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)

        # loop over the nodes deleting CPT
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes,cn)
            self.DeleteNodeTables(cnode)
        # make a streamer to the new cas file
        new_cas_streamer = self.NewFileStreamer(newCaseFile)

        if EMflag:
            print 'Learning new CPTs using EM algorithm'
            # to use EM learning, must first make a learner and set a couple options
            newlearner = self.NewLearner(pnC.learn_method_bn_const.EM_LEARNING)
            self.SetLearnerMaxTol(newlearner,1.0e-6)
            self.SetLearnerMaxIters(newlearner,1000)
            # now must associate the casefile with a caseset (weighted by unity)
            newcaseset = self.NewCaseset('currcases')
            self.AddFileToCaseset(newcaseset,new_cas_streamer,1.0)
            self.LearnCPTs(newlearner,allnodes,newcaseset,voodooPar)
            self.DeleteCaseset(newcaseset)
            self.DeleteLearner(newlearner)

        else:
            print 'Learning new CPTs using ReviseCPTsByCaseFile'
            self.ReviseCPTsByCaseFile(new_cas_streamer,allnodes,voodooPar)
        outfile_streamer = self.NewFileStreamer(outfilename)
        self.CompileNet(cnet)


        outfile_streamer = self.NewFileStreamer(outfilename)
        print 'Writing new net to: %s' %(outfilename)
        self.WriteNet(cnet,outfile_streamer)
        self.DeleteStream(outfile_streamer)  
        self.DeleteNet(cnet)

    def OpenNeticaNet(self,netName):
        '''
        Open a net identified by netName.
        Returns a pointer to the opened net after it is compiled
        '''
        # meke a streamer to the Net file
        cname = netName
        if '.neta' not in netName:
            cname += '.neta'
        net_streamer = self.NewFileStreamer(cname)
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)  
        self.CompileNet(cnet)      
        return cnet

    def ReadNodeInfo(self,netName):
        '''
        Read in all information on beliefs, states, and likelihoods for all 
        nodes in the net called netName
        '''
        # open the net stored in netName
        cnet = self.OpenNeticaNet(netName)
        #get the nodes and their number
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)
        print 'Reading Node information from net --> %s' %(netName)
        cNETNODES = dict()
        # loop over the nodes
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes,cn)
            cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
            cNETNODES[cnodename] =nodestruct()
            cNETNODES[cnodename].name = cth.c_char_p2str(self.GetNodeName(cnode))
            cNETNODES[cnodename].title = cth.c_char_p2str(self.GetNodeTitle(cnode))
            print '   Parsing node --> %s' %(cNETNODES[cnodename].title)
            cNETNODES[cnodename].Nbeliefs = self.GetNodeNumberStates(cnode)
            cNETNODES[cnodename].beliefs = cth.c_float_p2float(
                self.GetNodeBeliefs(cnode),
                cNETNODES[cnodename].Nbeliefs)
            cNETNODES[cnodename].likelihood = cth.c_float_p2float(
                self.GetNodeLikelihood(cnode),
                cNETNODES[cnodename].Nbeliefs)
            cNETNODES[cnodename].levels =  cth.c_double_p2float(
                self.GetNodeLevels(cnode),
                cNETNODES[cnodename].Nbeliefs + 1)

            # loop over the states in each node
            for cs in range(cNETNODES[cnodename].Nbeliefs):
                cNETNODES[cnodename].state.append(statestruct())
                cNETNODES[cnodename].state[-1].name = cth.c_char_p2str(
                    self.GetNodeStateName(cnode,cs))   
       
        self.DeleteNet(cnet)
        return cNETNODES
    
    def NodeParentIndexing(self,netName,casfile):
        '''
        Find all the configurations of states in the parent nodes for each response node
        '''
        # open the net stored in netName
        cnet = self.OpenNeticaNet(netName)
        #get the nodes and their number
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)

        #parent indices dictionary for the results
        parent_indices = dict()
        # now focus in on the repsonse nodes only
        respnodes = self.probpars.scenario.response
        for cr in respnodes:
            parent_indices[cr] = parent_inds()
            crespnode = self.GetNodeNamed(cr,cnet)
            # get the parent nodes and their names
            cparents = self.GetNodeParents(crespnode)
            numparents = self.LengthNodeList(cparents)
            for cp in np.arange(numparents):
                tmpnode = self.NthNode(cparents,cp)
                parent_indices[cr].parent_names.append(
                    cth.c_char_p2str(self.GetNodeName(tmpnode)))            
            
        # open a streamer to the CAS file we will read over
        cas_streamer = self.NewFileStreamer(casfile)
        
        # loop over the cases
        for ccas in np.arange(self.N):
            if ccas==0:
                case_posn = pnC.netica_const.FIRST_CASE
            else:
                case_posn = pnC.netica_const.NEXT_CASE
            # first set the findings according to what's in the case file
            case_posn_out = self.ReadNetFindings2(case_posn,cas_streamer,allnodes)
            # now, for each parent, in order, read the states
            for cr in respnodes:
                tmpinds = list()
                for cp in parent_indices[cr].parent_names:
                    cnode = self.GetNodeNamed(cp,cnet)
                    tmpinds.append(self.GetNodeFinding(cnode))
                parent_indices[cr].parent_indices.append(tmpinds)
        parent_indices[cr].parent_indices = np.array(
                parent_indices[cr].parent_indices,dtype=int)
        # clean up the temporary streamer and net
        self.DeleteNet(cnet)
        self.DeleteStream(cas_streamer)
        self.parent_inds = parent_indices
    

    def PredictBayesNeticaCV(self,cfold,cnetname,calval):
        '''
        function using Netica built-in testing functionality to evaluate Net
        '''
        ctestresults = netica_test()
        
        # open up the current net
        cnet = self.OpenNeticaNet(cnetname)
        #retract all the findings
        self.RetractNetFindings(cnet)
        
        # first create a caseset with the current leftout indices casefile
        if cfold > -10:
            if calval.upper() == 'CAL':
                ccasefile = '%s_fold_%d.cas' %(self.probpars.scenario.name,cfold)
            elif calval.upper()=='VAL':            
                ccasefile = '%s_fold_%d_leftout.cas' %(self.probpars.scenario.name,cfold)
            else:
                pass
        # unless this is the base case -->
        else:
            ccasefile = self.probpars.baseCAS
        currcases = self.NewCaseset('cval%d' %(np.abs(cfold)))        
        ccaseStreamer = self.NewFileStreamer(ccasefile)
        self.AddFileToCaseset(currcases,ccaseStreamer,100.0)
            
        # create a set of prediction nodes
        numprednodes = len(self.probpars.scenario.response)
        cnodelist = self.NewNodeList2(numprednodes,cnet)
        for i,cn in enumerate(self.probpars.scenario.response):
            cnode = self.GetNodeNamed(cn,cnet)
            self.SetNthNode(cnodelist,i,cnode)
        # create a tester object
        ctester = self.NewNetTester(cnodelist,cnodelist)
        self.DeleteNodeList(cnodelist)
        
        # test the network using the left-out cases
        # first retract all the findings and compile the net
        self.TestWithCaseset(ctester,currcases)
        self.DeleteCaseset(currcases)
        #
        # now get the results
        #
        ctestresults.logloss = dict()
        ctestresults.errrate = dict()
        ctestresults.quadloss = dict()
        ctestresults.confusion_matrix = dict()
        ctestresults.experience = dict()
        for cn in self.probpars.scenario.response:
            cnode = self.GetNodeNamed(cn,cnet)
            # get log loss
            ctestresults.logloss[cn] =  self.GetTestLogLoss(ctester,cnode)
            print 'LogLoss for %s --> %f' %(cn,ctestresults.logloss[cn])           
            # get error rate
            ctestresults.errrate[cn] = self.GetTestErrorRate(ctester,cnode)
            print 'ErrorRate for %s --> %f' %(cn,ctestresults.errrate[cn]) 
            # get quadratic loss
            ctestresults.quadloss[cn] = self.GetTestQuadraticLoss(ctester,cnode)
            print 'QuadLoss for %s --> %f' %(cn,ctestresults.quadloss[cn])    
            # write confusion matrix --- only for the base case
            if cfold < 0:
                print 'Calculating confusion matrix for %s' %(cn)
                ctestresults.confusion_matrix[cn] = self.ConfusionMatrix(ctester,cnode)
                # also calculate the experience for the node
                print 'Calculating Experience for the base Net, node --> %s' %(cn)
                ctestresults.experience[cn] = self.ExperienceAnalysis(cn,cnet)
            
        self.DeleteNetTester(ctester)
        self.DeleteNet(cnet)
        
        # write to the proper dictionary
        if cfold > -10:
            if calval.upper() == 'CAL':
                self.NeticaTests['CAL'].append(ctestresults)
            elif calval.upper() == 'VAL':
                self.NeticaTests['VAL'].append(ctestresults)
            else:
                pass
        else:
            self.BaseNeticaTests = ctestresults            
        
        
    def ExperiencePostProc(self):
        print "Post-Processing Experience data, matching with predicted nodes and cases"
        for cn in self.probpars.scenario.response:
            for ccas in np.arange(self.N):
                testinds = self.parent_inds[cn].parent_indices[ccas,:]
                tmp = testinds-self.BaseNeticaTests.experience[cn].parent_states                
                tmp = np.sum(np.abs(tmp),axis=1)
                cind = np.where(tmp==0)
                self.BaseNeticaTests.experience[cn].case_experience.append(
                    self.BaseNeticaTests.experience[cn].node_experience[cind[0]])
                
    
    
    def predictBayes(self,netName,N,casdata):
        '''
        netName --> name of the built net to make predictions on
        '''
        # first read in the information about a Net's nodes
        cNETNODES = self.ReadNodeInfo(netName)
        '''
        Initialize output 
        '''       
        # initialize dictionary of predictions objects
        cpred = dict()

        print 'Making predictions for net named --> %s' %(netName)
        cnet = self.OpenNeticaNet(netName)
        #retract all the findings
        self.RetractNetFindings(cnet)
        for CN in cNETNODES:
            CNODES = cNETNODES[CN]
            Cname = CNODES.name
            if Cname in self.probpars.scenario.response:
                cpred[Cname] = predictions()
                cpred[Cname].stats = pred_stats()
                Nbins = CNODES.Nbeliefs
                cpred[Cname].pdf = np.zeros((N,Nbins))
                cpred[Cname].ranges = np.array(CNODES.levels)
                # get plottable ranges
                if Nbins < len(CNODES.levels):
                    # continuos, so plot bin centers
                    CNODES.continuous = True
                    cpred[Cname].continuous = True
                    cpred[Cname].rangesplt = (cpred[Cname].ranges[1:]-
                                              0.5*np.diff(cpred[Cname].ranges))
                else:
                    #discrete so just use the bin values
                    cpred[Cname].rangesplt = cpred[Cname].ranges.copy()

                cpred[Cname].priorPDF = CNODES.beliefs

        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)
        #
        # Now loop over each input and get the Netica predictions
        #
        for i in np.arange(N):
            # first have to enter the values for each node
            # retract all the findings again
            self.RetractNetFindings(cnet)
            for cn in np.arange(numnodes):
                cnode = self.NthNode(allnodes,cn)
                cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
                # set the current node values
                if cnodename in self.probpars.scenario.nodesIn:
                    self.EnterNodeValue(cnode,casdata[cnodename][i])
            for cn in np.arange(numnodes):
            # obtain the updated beliefs from ALL nodes including input and output
                cnode = self.NthNode(allnodes,cn)
                cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
                if cnodename in self.probpars.scenario.response:
                    # get the current belief values
                    cpred[cnodename].pdf[i,:] = cth.c_float_p2float(
                        self.GetNodeBeliefs(cnode),
                        self.GetNodeNumberStates(cnode))
        #
        # Do some postprocessing for just the output nodes
        #
        currstds = np.ones((N,1))*1.0e-16
        for i in self.probpars.scenario.response:
            print 'postprocessing output node --> %s' %(i)
            # record whether the node is continuous or discrete
            if cpred[i].continuous:
                curr_continuous='continuous'
            else:
                curr_continuous = 'discrete'
            pdfRanges = cpred[i].ranges
            cpred[i].z = np.atleast_2d(casdata[i]).T
            pdfParam = np.hstack((cpred[i].z,currstds))
            pdfData = statfuns.makeInputPdf(pdfRanges,pdfParam,'norm',curr_continuous)

            cpred[i].probModelUpdate = np.nansum(pdfData * cpred[i].pdf,1)
            cpred[i].probModelPrior = np.nansum(pdfData * np.tile(cpred[i].priorPDF,
                                                                      (N,1)),1)
            cpred[i].logLikelihoodRatio = (np.log10(cpred[i].probModelUpdate + np.spacing(1)) - 
                                               np.log10(cpred[i].probModelPrior + np.spacing(1)))
            cpred[i].dataPDF = pdfData.copy()
            # note --> np.spacing(1) is like eps in MATLAB
            # get the PDF stats here
      
            print 'getting stats'
            cpred = self.PDF2Stats(i,cpred,alpha=0.1)

        self.DeleteNet(cnet)
        return cpred,cNETNODES
    
    def PDF2Stats(self,nodename, cpred, alpha = None):
        '''
        extract statistics from the PDF informed by a Bayesian Net

        most information is contained in self which is a pynetica object
        however, the nodename indicates which node to calculate stats for
        '''

        # normalize the PDF in case it doesn't sum to unity        
        pdf = np.atleast_2d(cpred[nodename].pdf).copy()
        pdf /= np.tile(np.atleast_2d(np.sum(pdf,1)).T,(1, pdf.shape[1]))

        # Start computing the statistics
        [Nlocs,Npdf] = pdf.shape
        blank = 0.0 + ~np.isnan(pdf[:,0])
        blank[blank==0]=np.nan
        blank = np.atleast_2d(blank).T

        # handle the specific case of a user-specified percentile range
        if alpha:
            self.alpha = alpha
            dalpha = (1.0 - alpha)/2.0
            # first return the percentile requested in bAlpha
            cpred[nodename].stats.palpha = blank*statfuns.getPy(
                alpha,pdf,
                cpred[nodename].ranges)

            # now get the tails from requested bAlpha
            # upper tail
            cpred[nodename].stats.palphaPlus = blank*statfuns.getPy(
                1.0-dalpha,pdf,
                cpred[nodename].ranges)

            # lower tail
            cpred[nodename].stats.palphaMinus = blank*statfuns.getPy(
                dalpha,pdf,
                cpred[nodename].ranges)
        # now handle the p75,p95, and p975 cases
        # 75th percentiles
        cpred[nodename].stats.p25 = blank*statfuns.getPy(0.25,pdf,
                                                             cpred[nodename].ranges)    
        cpred[nodename].stats.p75 = blank*statfuns.getPy(0.75,pdf,
                                                             cpred[nodename].ranges)    
        # 95th percentiles
        cpred[nodename].stats.p05 = blank*statfuns.getPy(0.05,pdf, 
                                                             cpred[nodename].ranges)    
        cpred[nodename].stats.p95 = blank*statfuns.getPy(0.95,pdf,
                                                             cpred[nodename].ranges)    
        # 97.5th percentiles
        cpred[nodename].stats.p025 = blank*statfuns.getPy(0.025,pdf, 
                                                              cpred[nodename].ranges)    
        cpred[nodename].stats.p975 = blank*statfuns.getPy(0.975,pdf,
                                                              cpred[nodename].ranges)  
        # MEDIAN  
        cpred[nodename].stats.median = blank*statfuns.getPy(0.5,pdf,
                                                                cpred[nodename].ranges)  

        # now get the mean, ML, and std values
        (cpred[nodename].stats.mean,
         cpred[nodename].stats.std,
         cpred[nodename].stats.mostProb) = statfuns.getMeanStdMostProb(pdf,
                                                                           cpred[nodename].ranges,
                                                                           cpred[nodename].continuous,
                                                                           blank)

        cpred[nodename].stats.skMean = statfuns.LSQR_skill(
            cpred[nodename].stats.mean,
            cpred[nodename].z-np.mean(cpred[nodename].z))

        cpred[nodename].stats.skML = statfuns.LSQR_skill(
            cpred[nodename].stats.mostProb,
            cpred[nodename].z-np.mean(cpred[nodename].z))
        Mresid = (cpred[nodename].stats.mean -
                  cpred[nodename].z)
        cpred[nodename].stats.rmseM = (
            np.sqrt(nanmean(Mresid**2)))  
        cpred[nodename].stats.meaneM = nanmean(Mresid)    
        MLresid = (cpred[nodename].stats.mostProb -
                   cpred[nodename].z)
        cpred[nodename].stats.rmseML = (
            np.sqrt(nanmean(MLresid**2)))  
        cpred[nodename].stats.meaneML = nanmean(MLresid)   
        cpred[nodename].stats.meanabserrM = nanmean(np.abs(Mresid))
        cpred[nodename].stats.meanabserrML = nanmean(np.abs(MLresid))
        
        
        return cpred
    
    def PredictBayesPostProc(self,cpred,outname,casname,cNeticaTestStats):
        ofp = open(outname,'w')
        ofp.write('Validation statistics for net --> %s and casefile --> %s\n'
                  %(outname,casname))   
        ofp.write('%14s '*12
                  %('Response','skillMean','rmseMean','meanErrMean','meanAbsErrMean',
                    'skillML','rmseML','meanErrML','meanAbsErrML','LogLoss','ErrorRate','QuadraticLoss')
                  + '\n')
        for i in self.probpars.scenario.response:
            print 'writing output for --> %s' %(i)
            ofp.write('%14s %14.4f %14.6e %14.6e %14.6e %14.4f %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n'
                      %(i,cpred[i].stats.skMean,
                        cpred[i].stats.rmseM,
                        cpred[i].stats.meaneM,
                        cpred[i].stats.meanabserrM,                        
                        cpred[i].stats.skML,
                        cpred[i].stats.rmseML,
                        cpred[i].stats.meaneML,
                        cpred[i].stats.meanabserrML,
                        cNeticaTestStats.logloss[i],
                        cNeticaTestStats.errrate[i],
                        cNeticaTestStats.quadloss[i]))
        ofp.close()
        outfileConfusion = re.sub('base_stats','Confusion',outname)
        ofpC = open(outfileConfusion,'w')
        ofpC.write('Confusion matrices for net --> %s and casefile --> %s\n' 
                  %(outname,casname))   
        for j in self.probpars.scenario.response:
            ofpC.write('*'*16 + '\nConfusion matrix for %s\n' %(j) + '*'*16 + '\n')
            numstates = len(self.NETNODES[j].levels)-1
            ofpC.write('%24s' %(''))
            for i in np.arange(numstates):
                ofpC.write('%24s' %('%8.4e--%8.4e'%(self.NETNODES[j].levels[i],
                                        self.NETNODES[j].levels[i+1])))
            ofpC.write('\n')
            for i in np.arange(numstates):
                ofpC.write('%24s' %('%8.4e--%8.4e'%(self.NETNODES[j].levels[i],
                                        self.NETNODES[j].levels[i+1])))
                for k in cNeticaTestStats.confusion_matrix[j][i,:]:
                    ofpC.write('%24d' %(k))
                ofpC.write('\n')
            ofpC.write('\n' * 2)
        ofpC.close()
            
            
        
        
    def PredictBayesPostProcCV(self,cpred,numfolds,ofp,calval,cNeticaTestStats):
        for cfold in np.arange(numfolds):
            for j in self.probpars.scenario.response:
                print 'writing %s cross-validation output for --> %s' %(calval,j)
                ofp.write('%14d %14s %14.4f %14.6e %14.6e %14.6e %14.4f %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n'
                      %(cfold,j,cpred[cfold][j].stats.skMean,
                        cpred[cfold][j].stats.rmseM,
                        cpred[cfold][j].stats.meaneM,
                        cpred[cfold][j].stats.meanabserrM,
                        cpred[cfold][j].stats.skML,
                        cpred[cfold][j].stats.rmseML,
                        cpred[cfold][j].stats.meaneML,
                        cpred[cfold][j].stats.meanabserrML,
                        cNeticaTestStats[cfold].logloss[j],
                        cNeticaTestStats[cfold].errrate[j],
                        cNeticaTestStats[cfold].quadloss[j]))
        

    def ConfusionMatrix(self,ctester,cnode):
        '''
        Makes a confusion matrix for a particular node specified by name in cnode
        within the tester environment laid out in ctester
        '''
        numstates = self.GetNodeNumberStates(cnode)
        
        confusion_matrix = np.zeros((numstates,numstates))
        for a in np.arange(numstates):
            for p in np.arange(numstates):
                confusion_matrix[a,p] = self.GetTestConfusion(ctester,cnode,p,a)
        return confusion_matrix

    def ExperienceAnalysis(self,cn,cnet):
        '''
        calculate the experience for the node named in cn
        '''
        cnex = experience()
        # get a list of the parents of the node
        testnode = self.GetNodeNamed(cn,cnet)
        #start a list for the cartesian sum of node states
        allstates = list()
        cparents = self.GetNodeParents(testnode)    
        numnodes = self.LengthNodeList(cparents)
        for cp in np.arange(numnodes):
            # append the name to the list of returned names
            cnode = self.NthNode(cparents,cp)
            cnex.parent_names.append(cth.c_char_p2str(self.GetNodeName(cnode)))
            # find the number of states for each parent
            allstates.append(np.arange(self.GetNodeNumberStates(
                self.NthNode(cparents,cp))))
        if numnodes > 1:
            cnex.parent_states = self.cartesian(allstates)
        else:
            cnex.parent_states = allstates
        for cs in cnex.parent_states:
            cnex.node_experience.append(self.GetnodeExperience(
                testnode,cs.ctypes.data_as(ct.POINTER(ct.c_int))))
        cnex.node_experience = np.array(cnex.node_experience)
        # change the null pointers (meaning 
        cnex.node_experience[cnex.node_experience<1]=0.0
        
        return cnex

        
        
    def SensitivityAnalysis(self):
        '''
        Peforms sensitivity analysis on each response node assuming all 
        input nodes are active (as defined in self.probpars.scenario)
        
        Reports results to a text file.
        '''
        print '\n' * 3 + '*' * 10 + '\n' + 'Performing Sensitity Analysis\n' + '*'*10
        # meke a streamer to the Net file
        net_streamer = self.NewFileStreamer(self.probpars.baseNET)
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)  
        self.CompileNet(cnet)              
        self.sensitivityvar = dict()
        self.sensitivityEntropy = dict()
        self.sensitivityEntropyNorm = dict()
        self.percentvarreduction = dict()
        allnodes = list()
        allnodes.extend(self.probpars.scenario.nodesIn)
        allnodes.extend(self.probpars.scenario.response)
        for cres in self.probpars.scenario.response:
            print "Calculating sensitivity to node --> %s" %(cres)
            # calculate the sensitivity for each response variable using all nodes  as Vnodes
            Qnode = self.GetNodeNamed(cres,cnet)
            Vnodes = self.GetNetNodes(cnet)
            self.sensitivityvar[cres] = dict()
            self.sensitivityEntropy[cres] = dict()
            self.sensitivityEntropyNorm[cres] = dict()            
            self.percentvarreduction[cres] = dict()
            sensvar = self.NewSensvToFinding(Qnode,Vnodes,ct.c_int(pnC.netica_const.VARIANCE_OF_REAL_SENSV))
            sensmutual = self.NewSensvToFinding(Qnode,Vnodes,ct.c_int(pnC.netica_const.ENTROPY_SENSV))
            for cn in allnodes:
                Vnode = self.GetNodeNamed(cn,cnet)
                self.sensitivityvar[cres][cn] = self.GetVarianceOfReal(sensvar,Vnode)
                self.sensitivityEntropy[cres][cn] = self.GetMutualInfo(sensmutual,Vnode)
                # percent variance reduction is the variance reduction of a node divided by variance reduction of self
            for cn in allnodes:
                self.percentvarreduction[cres][cn] = self.sensitivityvar[cres][cn]/self.sensitivityvar[cres][cres]
                self.sensitivityEntropyNorm[cres][cn] = self.sensitivityEntropy[cres][cn]/self.sensitivityEntropy[cres][cres]
            print "Deleting sensitivity to --> %s" %(cres)
            self.DeleteSensvToFinding(sensvar)
            self.DeleteSensvToFinding(sensmutual)
        self.DeleteNet(cnet)
        
        # #### WRITE OUTPUT #### #
        ofp = open(self.probpars.scenario.name + 'Sensitivity.dat','w')
        ofp.write('Sensitivity analysis for scenario --> %s\n' %(self.probpars.scenario.name))
        ofp.write('Base Case Net: %s\nBase Case Casfile: %s\n' %(self.probpars.baseNET,self.probpars.baseCAS))
        # write out the raw variance reduction values        
        ofp.write('#'*10 + '   Raw Variance Reduction Values   ' + '#'*10 + '\n')
        ofp.write('%-14s' %('Response_node '))
        for cn in allnodes:
            ofp.write('%-14s' %(cn))
        ofp.write('\n')
        for cres in self.sensitivityvar:
            ofp.write('%-14s' %(cres))
            for cn in allnodes:
                ofp.write('%-14.5f' %(self.sensitivityvar[cres][cn]))
            ofp.write('\n')
        # write out the percent variance reduction values                    
        ofp.write('#'*10 + '   Percent Variance Reduction Values   ' + '#'*10 + '\n')
        ofp.write('%-14s' %('Response_node '))
        for cn in allnodes:
            ofp.write('%-14s' %(cn))
        ofp.write('\n')
        for cres in self.percentvarreduction:
            ofp.write('%-14s' %(cres))
            for cn in allnodes:
                ofp.write('%-14.5f' %(self.percentvarreduction[cres][cn]*100.0))
            ofp.write('\n')
        # write out the mutual information (Entropy) values
        ofp.write('#'*10 + '   Mutual Information (Entropy)   ' + '#'*10 + '\n')
        ofp.write('%-14s' %('Response_node '))
        for cn in allnodes:
            ofp.write('%-14s' %(cn))
        ofp.write('\n')
        for cres in self.sensitivityEntropy:
            ofp.write('%-14s' %(cres))
            for cn in allnodes:
                ofp.write('%-14.5f' %(self.sensitivityEntropy[cres][cn]))
            ofp.write('\n')

        # write out the normalized mutual information (Entropy) values
        ofp.write('#'*10 + '   Mutual Information (Entropy) Normalized  ' + '#'*10 + '\n')
        ofp.write('%-14s' %('Response_node '))
        for cn in allnodes:
            ofp.write('%-14s' %(cn))
        ofp.write('\n')
        for cres in self.sensitivityEntropyNorm:
            ofp.write('%-14s' %(cres))
            for cn in allnodes:
                ofp.write('%-14.5f' %(self.sensitivityEntropyNorm[cres][cn]))
            ofp.write('\n')
        ofp.close()  
        
        
        
    def read_cas_file(self,casfilename):
        '''
        function to read in a casfile into a pynetica object.
        '''
        # first read in and strip out all comments and write out to a scratch file
        tmpdat = open(casfilename,'r').readlines()
        ofp = open('###tmp###','w')
        for line in tmpdat:
            #line = re.sub('\?','*',line)
            if '//' not in line:
                ofp.write(line)
            elif line.strip().split()[0].strip() == '//':
                pass
            elif '//' in line:
                line = re.sub('//.*','',line)
                if len(line.strip()) > 0:
                    ofp.write(line)
        ofp.close()                
        self.casdata = np.genfromtxt('###tmp###',names=True,
                                     dtype=None,missing_values = '*,?')
        os.remove('###tmp###')
        self.N = len(self.casdata)
        
    # cross validation driver
    def cross_val_setup(self):
        # open a file pointer to the stats output file for all the folds
        kfoldOFP_Val = open('%s_kfold_stats_VAL_%d_folds.dat' %(self.probpars.scenario.name,self.probpars.numfolds),'w')
        kfoldOFP_Val.write('Validation statistics for cross validation.\nBase net --> %s and casefile --> %s\n'
                  %(self.probpars.baseNET,self.probpars.baseCAS) + 'Current scenario is: %s\n' %(self.probpars.scenario.name))   
        kfoldOFP_Val.write('%14s '*13
                  %('Current_Fold','Response','skillMean','rmseMean','meanErrMean','meanAbsErrMean',
                    'skillML','rmseML','meanErrML','meanAbsErrML','LogLoss','ErrorRate','QuadraticLoss')
                  + '\n')
        
        kfoldOFP_Cal = open('%s_kfold_stats_CAL_%d_folds.dat' %(self.probpars.scenario.name,self.probpars.numfolds),'w')
        kfoldOFP_Cal.write('Calibration statistics for cross validation.\nBase net --> %s and casefile --> %s\n'
                  %(self.probpars.baseNET,self.probpars.baseCAS) + 'Current scenario is: %s\n' %(self.probpars.scenario.name))   
        kfoldOFP_Cal.write('%14s '*13
                %('Current_Fold','Response','skillMean','rmseMean','meanErrMean','meanAbsErrMean',
                  'skillML','rmseML','meanErrML','meanAbsErrML','LogLoss','ErrorRate','QuadraticLoss')
                + '\n')

        for cfold in np.arange(self.probpars.numfolds):
            self.allfolds.calNODES.append(None)
            self.allfolds.valNODES.append(None)
            self.allfolds.calpred.append(None)
            self.allfolds.valpred.append(None)
                                    
            cname = '%s_fold_%d.cas' %(self.probpars.scenario.name,cfold)
            self.allfolds.casfiles.append(cname)
            retinds = np.array(self.allfolds.retained[cfold],dtype=int)
            # outdat only includes the columns that are in CASheader
            outdat = np.atleast_2d(self.casdata[self.probpars.CASheader[0]][retinds]).T
            # caldata and valdata both include all columns for simplicity
            self.allfolds.caldata.append(self.casdata[retinds])
            leftoutinds = np.array(self.allfolds.leftout[cfold],dtype=int)
            # we will also make a CAS file for the leftout data for using Netica's testing codes
            outdatLeftOut = np.atleast_2d(self.casdata[self.probpars.CASheader[0]][leftoutinds]).T            
            self.allfolds.valdata.append(self.casdata[leftoutinds])
            self.allfolds.valN.append(len(leftoutinds))
            self.allfolds.calN.append(len(retinds))
            # concatenate together the columns of data that will make up the CAS files
            for i,chead in enumerate(self.probpars.CASheader):
                if i>0:
                    outdat = np.hstack((outdat,np.atleast_2d(self.casdata[chead][retinds]).T))
                    outdatLeftOut = np.hstack((outdatLeftOut,np.atleast_2d(self.casdata[chead][leftoutinds]).T))
            # write out the retained casefile            
            ofp = open(cname,'w')
            for cnode in self.probpars.CASheader:
                ofp.write('%s ' %(cnode))
            ofp.write('\n')
            np.savetxt(ofp,outdat)
            ofp.close()
            
            # write out the leftout casefile for later use with the NEtica validation testing functions
            ofpLeftOut = open(cname[:-4] + '_leftout.cas','w')
            for cnode in self.probpars.CASheader:
                ofpLeftOut.write('%s ' %(cnode))
            ofpLeftOut.write('\n')
            np.savetxt(ofpLeftOut,outdatLeftOut)
            ofpLeftOut.close()
            ofpLeftOut.close()
        return kfoldOFP_Val,kfoldOFP_Cal

    ###################################
    # Key helper functions for Netica #   
    ###################################

    def NewNeticaEnviron(self):
        '''
        create a new Netica environment based on operating system
        '''
        # first access the .dll or .so
        try:
            if 'window' in platform.system().lower():
                self.n = ct.windll.Netica
            else:
                self.n = ct.cdll.LoadLibrary("./libnetica.so")
        except:
            raise(dllFail(platform.system()))
        #next try to establish an environment for Netica
        self.env = ct.c_void_p(self.n.NewNeticaEnviron_ns(self.license,None,None))
        # try to intialize Netica
        res = self.n.InitNetica2_bn(self.env, ct.byref(self.mesg))
        if res >= 0:
            print '\n'*2 + '#' * 40 + '\nOpening Netica:'
            print self.mesg.value
        else:
            raise(NeticaInitFail(res.value))    
        print 'Netica is open\n' + '#'*40 + '\n' * 2
        
    def CloseNetica(self):
        res = self.n.CloseNetica_bn(self.env, ct.byref(self.mesg))    
        if res >= 0:
            print "Closing Netica:"
            print self.mesg.value
        else:
            raise(NeticaCloseFail(res.value))    
        self.n = None
        
    def GetError(self, severity = pnC.errseverity_ns_const.ERROR_ERR, after = None):
        res = self.n.GetError_ns(self.env, severity, after)
        if res: return ct.c_void_p(res)
        else:   return None

    def ErrorMessage(self, error):
        return self.n.ErrorMessage_ns(error)

    # general error-checking function    
    def chkerr(self,err_severity = pnC.errseverity_ns_const.ERROR_ERR):
        if self.GetError(err_severity):
            exceptionMsg = ("\npythonNeticaUtils: \nError " + 
                            str(ct.cast(ct.c_void_p(self.ErrorMessage(self.GetError(err_severity))), ct.c_char_p).value))
            raise NeticaException(exceptionMsg)

    ################################################################
    # Small definitions and little functions in alphabetical order #  
    ################################################################   
    def AddFileToCaseset(self,caseset,streamer,degree):
        self.n.AddFileToCaseset_cs(caseset,streamer,ct.c_double(degree),None)
        self.chkerr()

       
    def cartesian(self,arrays,out=None):   
        '''
        function to calculate the Cartesian sum of multiple arrays.
        This is used to provide the permutations (odometer style) of all
        the possible parent states when calculating experience.
        See: http://stackoverflow.com/questions/1208118/
        using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        '''

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
    
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
    
        m = n / arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            # recursive?
            self.cartesian(arrays[1:], out=out[0:m,1:])
            for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
        return out

    def CompileNet(self, net):
        self.n.CompileNet_bn(net)
        self.chkerr()

    def CopyNet(self,oldnet, newnetname,options):
        newnet = self.n.CopyNet_bn(oldnet,newnetname,self.env,options)
        self.chkerr()
        return newnet

    def CopyNodes(self,oldnodes,newnet,options):
        newnodes = self.n.CopyNodes_bn(oldnodes,newnet,options)
        self.chkerr()
        return newnodes 

    def DeleteCaseset(self,caseset):
        self.n.DeleteCaseset_cs(caseset)
        self.chkerr()

    def DeleteLearner(self,newlearner):
        self.n.DeleteLearner_bn(newlearner)
        self.chkerr()

    def DeleteNet(self,cnet):
        self.n.DeleteNet_bn(cnet)
        self.chkerr()
    
    def DeleteNetTester(self,ctester):
        self.n.DeleteNetTester_bn(ctester)
        self.chkerr()

    def DeleteNodeTables(self,node):
        self.n.DeleteNodeTables_bn(node)
        self.chkerr()

    def DeleteNodeList(self,cnodes):
        self.n.DeleteNodeList_bn(cnodes)
        self.chkerr()

    def DeleteStream(self,cstream):
        self.n.DeleteStream_ns(cstream)
        self.chkerr()

    def DeleteSensvToFinding(self,sens):
        self.n.DeleteSensvToFinding_bn(sens)
        self.chkerr()

    def EnterFinding(self,cnode,cval):
        self.n.EnterFinding_bn(cnode,ct.c_double(cval))
        self.chkerr()

    def EnterNodeValue(self,cnode,cval):
        self.n.EnterNodeValue_bn(cnode,ct.c_double(cval))
        self.chkerr()
        
    def GetMutualInfo(self,sensentrop,Vnode):
        tmpNeticaFun = self.n.GetMutualInfo_bn
        tmpNeticaFun.restype=ct.c_double
        retvar = self.n.GetMutualInfo_bn(sensentrop,Vnode)
        self.chkerr()
        return retvar        

    def GetNetNodes(self,cnet):
        allnodes = self.n.GetNetNodes2_bn(cnet,None)
        self.chkerr()
        return allnodes

    def GetNodeBeliefs(self,cnode):
        beliefs = self.n.GetNodeBeliefs_bn(cnode)
        self.chkerr()
        return beliefs

    def GetNodeExpectedValue(self,cnode):
        std_dev = ct.c_double()
        # make a temporary function variable to be able to set the
        # return value
        tmpNeticaFun = self.n.GetNodeExpectedValue_bn
        tmpNeticaFun.restype=ct.c_double
        expected_val = tmpNeticaFun(cnode,ct.byref(std_dev),
                                    None,None)
        self.chkerr()
        return expected_val, std_dev.value

    def GetnodeExperience(self,cnode,parent_states):
        tmpNeticaFun = self.n.GetNodeExperience_bn
        tmpNeticaFun.restype=ct.c_double
        experience = tmpNeticaFun(cnode,parent_states)
        self.chkerr()
        return experience
    
    def GetNodeFinding(self,cnode):
        cf = self.n.GetNodeFinding_bn(cnode)
        self.chkerr()
        return cf
    
    def GetNodeLevels(self,cnode):
        nodelevels = self.n.GetNodeLevels_bn(cnode)
        self.chkerr()
        return nodelevels

    def GetNodeLikelihood(self,cnode):
        nodelikelihood = self.n.GetNodeLikelihood_bn(cnode)
        self.chkerr()
        return nodelikelihood

    def GetNodeName(self,cnode):
        cname = self.n.GetNodeName_bn(cnode)
        self.chkerr()
        return cname

    def GetNodeNamed(self,nodename,cnet):
        retnode = self.n.GetNodeNamed_bn(nodename,cnet)
        self.chkerr()
        return(retnode)
    
    def GetNodeNumberStates(self,cnode):
        numstates = self.n.GetNodeNumberStates_bn(cnode)
        self.chkerr()
        return numstates

    def GetNodeParents(self,cnode):
        parents = self.n.GetNodeParents_bn(cnode)
        self.chkerr()
        return parents
    
    def GetNodeStateName(self,cnode,cstate):
        stname = self.n.GetNodeStateName_bn(cnode,ct.c_int(cstate))
        self.chkerr()
        return stname

    def GetNodeTitle(self,cnode):
        ctitle = self.n.GetNodeTitle_bn(cnode)
        self.chkerr()
        return ctitle

    def GetTestLogLoss(self,ctester,cnode):
        tmpNeticaFun = self.n.GetTestLogLoss_bn
        tmpNeticaFun.restype=ct.c_double
        logloss = self.n.GetTestLogLoss_bn(ctester,cnode)
        self.chkerr()
        return logloss
    
    def GetTestConfusion(self,ctester,cnode,predState,actualState):
        tmpNeticaFun = self.n.GetTestConfusion_bn
        tmpNeticaFun.restype=ct.c_double        
        confusion = tmpNeticaFun(ctester,cnode,ct.c_int(predState),ct.c_int(actualState))
        self.chkerr()
        return confusion
    
    def GetTestErrorRate(self,ctester,cnode):
        tmpNeticaFun = self.n.GetTestErrorRate_bn
        tmpNeticaFun.restype=ct.c_double
        errrate = tmpNeticaFun(ctester,cnode)
        self.chkerr()
        return errrate
    
    def GetTestQuadraticLoss(self,ctester,cnode):
        tmpNeticaFun = self.n.GetTestQuadraticLoss_bn
        tmpNeticaFun.restype=ct.c_double
        quadloss = self.n.GetTestQuadraticLoss_bn(ctester,cnode)
        self.chkerr()
        return quadloss
        
    def GetVarianceOfReal(self,sensv,Vnode):
        tmpNeticaFun = self.n.GetVarianceOfReal_bn
        tmpNeticaFun.restype=ct.c_double
        retvar = self.n.GetVarianceOfReal_bn(sensv,Vnode)
        self.chkerr()
        return retvar
        
    def LearnCPTs(self,learner,nodes,caseset,voodooPar):
        self.n.LearnCPTs_bn(learner,nodes,caseset,ct.c_double(voodooPar))
        self.chkerr()

    def LengthNodeList(self,nodelist):
        res = self.n.LengthNodeList_bn(nodelist)
        self.chkerr()
        return res    

    def NewCaseset(self,name):
        newcaseset = self.n.NewCaseset_cs(name,self.env)
        self.chkerr()
        return newcaseset

    def NewFileStreamer(self,infile):
        streamer =  self.n.NewFileStream_ns (infile, self.env,None)
        self.chkerr()
        return streamer

    def NewLearner(self,method):
        newlearner = self.n.NewLearner_bn(method,None,self.env)
        self.chkerr()
        return newlearner

    def NewNet(self, netname):
        newnet = self.n.NewNet_bn(netname,self.env)
        self.chkerr()
        return newnet
    
    def NewNetTester(self,test_nodes,unobs_nodes):
        tester = self.n.NewNetTester_bn(test_nodes,unobs_nodes,ct.c_int(-1))
        self.chkerr()
        return tester

    def NewNodeList2(self,length,cnet):
        nodelist = self.n.NewNodeList2_bn(ct.c_int(length),cnet)
        self.chkerr()
        return nodelist
    
    def NewSensvToFinding(self,Qnode,Vnodes,what_find):
        sensv = self.n.NewSensvToFinding_bn(Qnode,Vnodes,what_find)
        self.chkerr()
        return sensv

    def NthNode(self,nodelist,index_n):
        cnode = self.n.NthNode_bn(nodelist,ct.c_int(index_n))
        self.chkerr()
        return cnode
        
    def ReadNet(self,streamer):
        cnet = self.n.ReadNet_bn(streamer,ct.c_int(pnC.netica_const.NO_WINDOW))
        # check for errors
        self.chkerr()
        # reset the findings
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()
        return cnet

    def ReadNetFindings2(self,case_posn,filestream,allnodes):
        #tmpneticafun = self.n.ReadNetFindings2_bn
        #tmpneticafun.restype = ct.c_long
        case_position = ct.c_long(case_posn)
        self.n.ReadNetFindings2_bn(ct.byref(case_position),
                            filestream,
                            ct.c_ubyte(0),
                            allnodes,None,None)
        self.chkerr()      
        return case_position.value
                                   
                                   
    def RetractNetFindings(self,cnet):
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()

    def ReviseCPTsByCaseFile(self,casStreamer,cnodes,voodooPar):
        self.n.ReviseCPTsByCaseFile_bn(casStreamer,cnodes,ct.c_int(0),
                                       ct.c_double(voodooPar))
        self.chkerr()

    def SetNetAutoUpdate(self,cnet,belief_value):
        self.n.SetNetAutoUpdate_bn(cnet,belief_value)
        self.chkerr()     
        
    def SetNthNode(self,nodelist,position,cnode):
        self.n.SetNthNode_bn(nodelist,ct.c_int(position),cnode)
        self.chkerr()

    def SetLearnerMaxIters(self,learner,maxiters):
        self.n.SetLearnerMaxIters_bn(learner,ct.c_int(maxiters))
        self.chkerr()    

    def SetLearnerMaxTol(self,learner,tol):
        self.n.SetLearnerMaxTol_bn(learner,ct.c_double(tol))
        self.chkerr()         
        
    def TestWithCaseset(self,test,cases):
        self.n.TestWithCaseset_bn(test,cases)
        self.chkerr()
        
    def WriteNet(self,cnet,filename_streamer):
        self.n.WriteNet_bn(cnet,filename_streamer)
        self.chkerr()

#################
# Error Classes #
#################
# -- can't open external file
class dllFail(Exception):
    def __init__(self,cplat):
        self.cplat = cplat
    def __str__(self):
        if "windows" in self.cplat.lower():
            return("\n\nCannot open Netica.dll.\nBe sure it's in the path")
        else:
            return("\n\nCannot open libnetica.so.\nBe sure it's in the path")
# -- can't initialize Netica
class NeticaInitFail(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return("\n\nCannot initialize Netica. Netica message is:\n%s\n" 
               %(self.msg))
# -- can't close Netica
class NeticaCloseFail(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return("\n\nCannot properly close Netica. Netica message is:\n%s\n" 
               %(self.msg))
# -- General Netica Exception
class NeticaException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg