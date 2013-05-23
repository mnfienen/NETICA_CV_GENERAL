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
        self.rmseM = None
        self.meaneM = None
        self.rmseML = None
        self.meaneML = None


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
        self.n = None #this is the netica environment
        self.mesg = ct.create_string_buffer('\000' * 1024)
        self.basepred = None
        
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
            print "Warning: License File [%s] not found.\n Opening Netica without licence, which will limit size of nets that can be used." %(self.licensefile)
            self.license = None         
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
        self.NewNeticaEnviron()
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
            cnode = self.NthNode(allnodes,ct.c_int(cn))
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
        self.DeleteNet(cnet)
        self.CloseNetica()

    def OpenNeticaNet(self,netName):
        '''
        Start a Netica environment and open a net identified by
        netName.
        Returns a pointed to the opened net after it is compiled
        '''
        # create a Netica environment
        self.NewNeticaEnviron()
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
        cNETNODES = []
        # loop over the nodes
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes,ct.c_int(cn))
            cNETNODES.append(nodestruct())
            cNETNODES[-1].name = cth.c_char_p2str(self.GetNodeName(cnode))
            cNETNODES[-1].title = cth.c_char_p2str(self.GetNodeTitle(cnode))
            print '   Parsing node --> %s' %(cNETNODES[-1].title)
            cNETNODES[-1].Nbeliefs = self.GetNodeNumberStates(cnode)
            cNETNODES[-1].beliefs = cth.c_float_p2float(
                self.GetNodeBeliefs(cnode),
                cNETNODES[-1].Nbeliefs)
            cNETNODES[-1].likelihood = cth.c_float_p2float(
                self.GetNodeLikelihood(cnode),
                cNETNODES[-1].Nbeliefs)
            cNETNODES[-1].levels =  cth.c_double_p2float(
                self.GetNodeLevels(cnode),
                cNETNODES[-1].Nbeliefs + 1)

            # loop over the states in each node
            for cs in range(cNETNODES[-1].Nbeliefs):
                cNETNODES[-1].state.append(statestruct())
                cNETNODES[-1].state[-1].name = cth.c_char_p2str(
                    self.GetNodeStateName(cnode,cs))            
        self.CloseNetica()
        return cNETNODES
    
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
        for CNODES in cNETNODES:
            Cname = CNODES.name
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
                cnode = self.NthNode(allnodes,ct.c_int(cn))
                cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
                # set the current node values
                if cnodename in self.probpars.scenario.nodesIn:
                    self.EnterNodeValue(cnode,casdata[cnodename][i])
            for cn in np.arange(numnodes):
            # obtain the updated beliefs from ALL nodes including input and output
                cnode = self.NthNode(allnodes,ct.c_int(cn))
                cnodename = cth.c_char_p2str(self.GetNodeName(cnode))
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
            if 'mean_DTW' in cpred.keys():
                print cpred['mean_DTW'].pdf[-1]
            print 'getting stats'
            cpred = self.PDF2Stats(i,cpred,alpha=0.1)


        self.CloseNetica()
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
        
        return cpred
    
    def PredictBayesPostProc(self,cpred,outname,casname):
        ofp = open(outname + 'stats.dat','w')
        ofp.write('Validation statistics for net --> %s and casefile --> %s\n'
                  %(outname,casname))   
        ofp.write('%14s '*7 
                  %('Response','skillMean','rmseMean','meanErrMean','skillML','rmseML','meanErrML')
                  + '\n')
        for i in self.probpars.scenario.response:
            print 'writing output for --> %s' %(i)
            ofp.write('%14s %14.4f %14.6e %14.6e %14.4f %14.6e %14.6e\n'
                      %(i,cpred[i].stats.skMean,
                        cpred[i].stats.rmseM,
                        cpred[i].stats.meaneM,
                        cpred[i].stats.skML,
                        cpred[i].stats.rmseML,
                        cpred[i].stats.meaneML))
        ofp.close()

    def PredictBayesPostProcCV(self,cpred,cfold,ofp,calval):
        for i in self.probpars.scenario.response:
            print 'writing %s cross-validation output for --> %s' %(calval,i)
            ofp.write('%14d %14s %14.4f %14.6e %14.6e %14.4f %14.6e %14.6e\n'
                      %(cfold,i,cpred[i].stats.skMean,
                        cpred[i].stats.rmseM,
                        cpred[i].stats.meaneM,
                        cpred[i].stats.skML,
                        cpred[i].stats.rmseML,
                        cpred[i].stats.meaneML))



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
        kfoldOFP_Val.write('%14s '*8 
                  %('Current Fold','Response','skillMean','rmseMean','meanErrMean','skillML','rmseML','meanErrML')
                  + '\n')
        
        kfoldOFP_Cal = open('%s_kfold_stats_CAL_%d_folds.dat' %(self.probpars.scenario.name,self.probpars.numfolds),'w')
        kfoldOFP_Cal.write('Calibration statistics for cross validation.\nBase net --> %s and casefile --> %s\n'
                  %(self.probpars.baseNET,self.probpars.baseCAS) + 'Current scenario is: %s\n' %(self.probpars.scenario.name))   
        kfoldOFP_Cal.write('%14s '*8 
                  %('Current Fold','Response','skillMean','rmseMean','meanErrMean','skillML','rmseML','meanErrML')
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
            self.allfolds.valdata.append(self.casdata[leftoutinds])
            self.allfolds.valN.append(len(leftoutinds))
            self.allfolds.calN.append(len(retinds))
            for i,chead in enumerate(self.probpars.CASheader):
                if i>0:
                    outdat = np.hstack((outdat,np.atleast_2d(self.casdata[chead][retinds]).T))
            ofp = open(cname,'w')
            for cnode in self.probpars.CASheader:
                ofp.write('%s ' %(cnode))
            ofp.write('\n')
            np.savetxt(ofp,outdat)
            ofp.close()
        return kfoldOFP_Val,kfoldOFP_Cal
    # general error-checking function    
    def chkerr(self,err_severity = pnC.errseverity_ns_const.ERROR_ERR):
        if self.GetError(err_severity):
            exceptionMsg = ("pythonNeticaUtils: Error in " + 
                            str(ct.cast(ct.c_void_p(self.ErrorMessage(self.GetError(err_severity))), ct.c_char_p).value))
            raise NeticaException(exceptionMsg)

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
            print 'Opening Netica:'
            print self.mesg.value
        else:
            raise(NeticaInitFail(res.value))    

    def CloseNetica(self):
        res = self.n.CloseNetica_bn(self.env, ct.byref(self.mesg))    
        if res >= 0:
            print "Closing Netica:"
            print self.mesg.value
        else:
            raise(NeticaCloseFail(res.value))    

    def GetError(self, severity = pnC.errseverity_ns_const.ERROR_ERR, after = None):
        res = self.n.GetError_ns(self.env, severity, after)
        if res: return ct.c_void_p(res)
        else:   return None

    def ErrorMessage(self, error):
        return self.n.ErrorMessage_ns(error)


        ################################################################
    # Small definitions and little functions in alphabetical order #  
    ################################################################   
    def AddFileToCaseset(self,caseset,streamer,degree):
        self.n.AddFileToCaseset_cs(caseset,streamer,ct.c_double(degree))
        self.chkerr()

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

    def DeleteNodeTables(self,node):
        self.n.DeleteNodeTables_bn(node)
        self.chkerr()

    def DeleteStream(self,cstream):
        self.n.DeleteStream_ns(cstream)
        self.chkerr()

    def EnterFinding(self,cnode,cval):
        self.n.EnterFinding_bn(cnode,ct.c_double(cval))
        self.chkerr()

    def EnterNodeValue(self,cnode,cval):
        self.n.EnterNodeValue_bn(cnode,ct.c_double(cval))
        self.chkerr()

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
        print expected_val
        print std_dev.value
        self.chkerr()
        return expected_val, std_dev.value

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

    def GetNodeNumberStates(self,cnode):
        numstates = self.n.GetNodeNumberStates_bn(cnode)
        self.chkerr()
        return numstates

    def GetNodeStateName(self,cnode,cstate):
        stname = self.n.GetNodeStateName_bn(cnode,ct.c_int(cstate))
        self.chkerr()
        return stname

    def GetNodeTitle(self,cnode):
        ctitle = self.n.GetNodeTitle_bn(cnode)
        self.chkerr()
        return ctitle

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

    def NthNode(self,nodelist,index_n):
        cnode = self.n.NthNode_bn(nodelist,index_n)
        self.chkerr()
        return cnode

    def ReadNet(self,streamer):
        cnet = self.n.ReadNet_bn(streamer,pnC.netica_const.NO_WINDOW)
        # check for errors
        self.chkerr()
        # reset the findings
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()
        return cnet

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

    def SetLearnerMaxIters(self,learner,maxiters):
        self.n.SetLearnerMaxIters_bn(learner,ct.c_int(maxiters))
        self.chkerr()    

    def SetLearnerMaxTol(self,learner,tol):
        self.n.SetLearnerMaxTol_bn(learner,ct.c_double(tol))
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