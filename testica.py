import pythonNeticaUtils as pyn
import pdb

netInst = pyn.pynetica()

netInst.read_lic_file('mikeppwd.txt')

netInst.NewNeticaEnviron()
netInst.CloseNetica()



