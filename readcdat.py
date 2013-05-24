import pickle, gzip
ifp = gzip.open('example2_cdat.pklz','rb')
cdat = pickle.load(ifp)
ifp.close()