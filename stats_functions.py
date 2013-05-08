import numpy as np
from scipy.stats import norm

def makeInputPdf(pdfRanges, pdfParam,pdfType='norm',cont_discrete='continuous'):
    '''
    pdfRanges --> gives the range for each bin in the node
    pdfParam --> a Nx2 vector of mean and std
    pdfType --> indicates the distribution assumption (must be 'norm' for now)
    cont_discrete --> indicates of node is 'continuous' or 'discrete'
                                (MUST BE CONTINUOUS FOR NOW!)
    
    returns PDF
    '''
    [N,m] = pdfParam.shape
    r = len(pdfRanges)
    if cont_discrete == 'continuous':
        PDF = np.zeros((N,r-1))
        for i in np.arange(N):
            cdf = norm.cdf(pdfRanges,pdfParam[i,0],pdfParam[i,1])
            cdf[-1]=1.0
            PDF[i,:] = np.diff(cdf)
    return PDF