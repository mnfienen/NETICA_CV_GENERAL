import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

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

def getPy(p,pdf,ranges):
    '''
    return the value of y with probability p from the PDF supplied
    this is the percentile (p)
    '''
    ranges = np.array(ranges)
    M = len(ranges)
    N,Nbins = pdf.shape
    f = np.cumsum(pdf,1)
    f = np.hstack((np.zeros((N,1)),f))
    # now make special cases for discrete or continuous
    if Nbins == M:
        # D I S C R E T E ((( NOT TESTED!!!!!)))
        if p<0.5: # indicates lower bound
            f = sum(f<p,1)
        else: # upper bound
            f = f>p
            f = M-sum(f,1)+1
            # catch any overrun
            f[f>M] = M
        py = ranges[f-1]
    else:
        # C O N T I N U O U S
        py = np.nan * np.ones((N,1))
        for i in np.arange(N):
            funique,uniqueid = np.unique(f[i,:],return_index=True)
            # note strange syntax in the next line because interp1d returns
            # a function that then performs the interpolation
            py[i] = interp1d(funique,ranges[uniqueid])(p)
    return py
