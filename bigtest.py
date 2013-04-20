import pythonNeticaUtils as pyn

cdat = pyn.pynetica()

cdat.read_cas_file('test.cas')

cdat.read_lic_file('mikeppwd.txt')

cdat.NewNeticaEnviron()

cdat.CloseNetica()



