import numpy as np

infile = 'glass_sqrt_dist_set4_kfold_stats_CAL_10_folds.dat'

header = open(infile,'r').readlines()[0:4]

stats = ['min','max','mean','median']

response_headers = ['skillMean',
                    'rmseMean',
                    'meanErrMean',
                    'meanAbsErrMean',
                    'skillML',
                    'rmseML',
                    'meanErrML',
                    'meanAbsErrML']


indat = np.genfromtxt(infile,skiprows=3,dtype=None, names=True)

unique_responses = np.unique(indat['Response'])
numfolds = np.max(indat['Current_Fold'])+1
outdat = dict() # dictionary of responses

for cres in unique_responses:
    outdat[cres]=dict()
    outdat[cres]['min'] = dict()
    outdat[cres]['max'] = dict()
    outdat[cres]['mean'] = dict()
    outdat[cres]['median'] = dict()
    currinds = np.where(indat['Response'] == cres)[0]
    for cstat in response_headers:
        outdat[cres]['min'][cstat] = np.min(indat[cstat][currinds])
        outdat[cres]['max'][cstat] = np.max(indat[cstat][currinds])
        outdat[cres]['mean'][cstat] = np.mean(indat[cstat][currinds])                        
        outdat[cres]['median'][cstat] = np.median(indat[cstat][currinds])
                    
ofp = open(infile[:-4] + '_SUMMARY.dat','w')
ofp.write('SUMMARY STATISTICS-->\n')
for line in header:
    ofp.write(line)
ofp.write('%16s%16s' %('Stat','Response'))
for chead in response_headers:
    ofp.write('%16s' %(chead))
ofp.write('\n')

for currstat in stats:
    for cresp in unique_responses:
        ofp.write('%16s' %(currstat))        
        ofp.write('%16s' %(cresp))
        for cval in response_headers:
            if 'skill' in cval:
                ofp.write('%16.5f' %(outdat[cresp][currstat][cval]))
            else:
                ofp.write('%16.5e' %(outdat[cresp][currstat][cval]))
        ofp.write('\n')
    ofp.write('\n')
ofp.close()
