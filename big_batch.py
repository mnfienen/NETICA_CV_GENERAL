import os

numfolds = 6
baseroot = 'SEEDA1A1B7D7E7'
basecases = ['log','sqrt','sqrt_dist','noxform']
allfiles = os.listdir(os.getcwd())
'''
for ccas in basecases:
    runline1 = 'python Divide_CAS.py %s%s.cas %d' %(baseroot,ccas,numfolds)
    print runline1
    os.system(runline1)
    runline2 = 'python XML_for_multiple_sets.py glass%s.xml %d' %(ccas,numfolds)
    print runline2
    os.system(runline2)

for cfile in allfiles:
    if cfile[-4:] == '.xml':
        runstring = 'python CV_driver.py %s' %(cfile)
        print runstring
        os.system(runstring)
''' 
for cfile in allfiles:
    if cfile[-4:] == '.dat' and ('CAL' in cfile or 'VAL' in cfile):
        runstring = 'python postproc_summarize_kfold.py %s' %(cfile)
        print runstring
        os.system(runstring)
    