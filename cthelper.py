import ctypes as ct

# convert a ctypes character pointer to a string
def c_char_p2str(cp):
    return ct.cast(cp,ct.c_char_p).value

# convert a ctypes float pointer to a float
# N.B. one can inadvertantly reference off the end of the resulting list!!\
# it is best to trim based on the known length of the returned array!
def c_float_p2str(cp):
    return ct.cast(cp,ct.POINTER(ct.c_float))
    
