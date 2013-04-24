import pythonNeticaUtils as pyn

cdat = pyn.pynetica('mikeppwd.txt')

cdat.read_cas_file('test.cas')

#cdat.NewNeticaEnviron()


cdat.rebuild_net('SLR.neta','test.cas',2,'SLR_new.neta')

cdat.CloseNetica()



