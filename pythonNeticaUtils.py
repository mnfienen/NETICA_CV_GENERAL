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
    def compile_net(self,NetName,newCaseFile,voodoPar,outfilename):
        '''
         compile_net(NetName,newCaseFilename,voodoPar,outfilename)
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
        streamer = n.NewFileStream_ns (NetName, self.env,None)
        cnet = n.ReadNet_bn(streamer,pnC.netica_const.NO_VISUAL_INFO)
        # check for errors
        while error=n.GetError_ns(self.env,pnC.errseverity_ns_const.XXX_ERROR,)
    
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
    def chkerr():
        if self.GetError(NE.ERROR_ERR):
	   exceptionMsg = "PyNetica: Error in " + 
	   str(ct.cast(ct.c_void_p(self.ErrorMessage(self.GetError(NE.ERROR_ERR))), ct.c_char_p).value)
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
    def ErrorMessage(self, error):
	return self.n.ErrorMessage_ns(error)
    
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