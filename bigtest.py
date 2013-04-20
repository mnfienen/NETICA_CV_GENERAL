import pythonNeticaUtils as pyn

cdat = pyn.pynetica('mikeppwd.txt')

cdat.read_cas_file('test.cas')

cdat.NewNeticaEnviron()

cdat.CloseNetica()



