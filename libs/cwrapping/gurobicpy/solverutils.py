import numpy as np

def generatecutzeroth(row):
    ###
    # generate cut that includes cost/obj row as well
    ###
    n = row.size
    a = row[1:n]
    b = row[0]
    cut_a = a - np.floor(a)
    cut_b = b - np.floor(b)
    return cut_a,cut_b

def roundmarrays(x,delta=1e-7):
    '''
	if certain components of x are very close to integers, round them
	'''
    index = np.where(abs(np.round(x)-x)<delta)
    x[index] = np.round(x)[index]
    return x

