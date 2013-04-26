import numpy as np
import os
import re
import ctypes as ct
import platform
import pythonNeticaConstants as pnC


class pynetica:
    def __init__(self,licfile):
        self.casdata = None
        self.n = None #this is the netica environment
        self.mesg = ct.create_string_buffer('\000' * 1024)
        # read in the license file information
        self.licensefile = licfile
        if os.path.exists(self.licensefile):
            self.license = open(self.licensefile,'r').readlines()[0].strip().split()[0]
        else:
            print "Warning: License File [%s] not found." %(self.licensefile)
            self.license = None 
    #####
    # Helper functions that drive Netica functions
    def rebuild_net(self,NetName,newCaseFile,voodooPar,outfilename):
        '''
         rebuild_net(NetName,newCaseFilename,voodooPar,outfilename)
         a m!ke@usgs joint <mnfienen@usgs.gov>
         function to build the CPT tables for a new CAS file on an existing NET
         (be existing, meaning that the nodes, edges, and bins are dialed)
         INPUT:
               NetName --> a filename, including '.neta' extension
               newCaseFilename --> new case file including '.cas' extension
               voodoPar --> the voodoo tuning parameter for building CPTs
               outfilename --> netica file for newly build net (including '.neta')
         '''   
         # create a Netica environment
        self.NewNeticaEnviron()
        # meke a streamer to the Net file
        net_streamer = self.NewFileStreamer(NetName)
        # read in the net using the streamer        
        cnet = self.ReadNet(net_streamer)
        # remove the input net streamer
        self.DeleteStream(net_streamer)
        # make a copy of the Net for output

        newnet = self.CopyNet(cnet,outfilename[:-5],"no_tables")
        self.CompileNet(newnet)
        # find the nodes
        newnodes = self.GetNetNodes(newnet)
        # make a streamer to the new cas file
        new_cas_streamer = self.NewFileStreamer(newCaseFile)
        self.ReviseCPTsByCaseFile(new_cas_streamer,newnodes,voodooPar)

        '''
        #find the names of the nodes
        allnodes = self.GetNetNodes(cnet)
        numnodes = self.LengthNodeList(allnodes)
        
        for cn in np.arange(numnodes):
            cnode = self.NthNode(allnodes,ct.c_int(cn))
            self.DeleteNodeTables(cnode)
        # make a streamer to the new cas file
        new_cas_streamer = self.NewFileStreamer(newCaseFile)
        self.ReviseCPTsByCaseFile(new_cas_streamer,allnodes,voodooPar)
        outfile_streamer = self.NewFileStreamer(outfilename)
        self.CompileNet(cnet)
        '''

        outfile_streamer = self.NewFileStreamer(outfilename)
        self.WriteNet(newnet,outfile_streamer)
        self.DeleteNet(cnet)
        self.DeleteNet(newnet)
        
            
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

    # general error-checking function    
    def chkerr(self,err_severity = pnC.errseverity_ns_const.ERROR_ERR):
        if self.GetError(err_severity):
	   exceptionMsg = ("pythonNeticaUtils: Error in " + 
	   str(ct.cast(ct.c_void_p(self.ErrorMessage(self.GetError(err_severity))), ct.c_char_p).value))
	   raise NeticaException(exceptionMsg)

    #############
    # Key uber functions for Netica    
        
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
    
    ###############
    # Small definitions and little functions
    def GetError(self, severity = pnC.errseverity_ns_const.ERROR_ERR, after = None):
		res = self.n.GetError_ns(self.env, severity, after)
		if res: return ct.c_void_p(res)
		else:   return None
		
    def ErrorMessage(self, error):
	return self.n.ErrorMessage_ns(error)
    def LengthNodeList(self,nodelist):
        res = self.n.LengthNodeList_bn(nodelist)
        self.chkerr()
        return res
        
    def NewFileStreamer(self,infile):
        streamer =  self.n.NewFileStream_ns (infile, self.env,None)
        self.chkerr()
        return streamer
        
    def DeleteNet(self,cnet):
        self.n.DeleteNet_bn(cnet)
        self.chkerr()
        
    def DeleteStream(self,cstream):
        self.n.DeleteStream_ns(cstream)
        self.chkerr()
        
    def SetNetAutoUpdate(self,cnet,belief_value):
        self.n.SetNetAutoUpdate_bn(cnet,belief_value)
        self.chkerr()
        
    def CompileNet(self, net):
        self.n.CompileNet_bn(net)
        self.chkerr()
        
    def CopyNet(self,oldnet, newnetname,options):
        newnet = self.n.CopyNet_bn(oldnet,newnetname,self.env,options)
        self.chkerr()
        return newnet
        
    def ReadNet(self,streamer):
        cnet = self.n.ReadNet_bn(streamer,pnC.netica_const.NO_WINDOW)
        # check for errors
        self.chkerr()
        # reset the findings
        self.n.RetractNetFindings_bn(cnet)
        self.chkerr()
        return cnet
        
    def NthNode(self,nodelist,index_n):
        cnode = self.n.NthNode_bn(nodelist,index_n)
        self.chkerr()
        return cnode
        
    def DeleteNodeTables(self,node):
        self.n.DeleteNodeTables_bn(node)
        self.chkerr()
        
    def GetNetNodes(self,cnet):
        allnodes = self.n.GetNetNodes2_bn(cnet,None)
        self.chkerr()
        return allnodes
        
    def ReviseCPTsByCaseFile(self,casStreamer,cnodes,voodooPar):
        self.n.ReviseCPTsByCaseFile_bn(casStreamer,cnodes,ct.c_int(0),
                                                ct.c_float(voodooPar))
        self.chkerr()
        
    def NewNet(self, netname):
        self.n.NewNet_bn(netname,self.env)
        self.chkerr()
        
    def WriteNet(self,cnet,filename_streamer):
        self.n.WriteNet_bn(cnet,filename_streamer)
        self.chkerr()
        
###################
# Error Classes
###################
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