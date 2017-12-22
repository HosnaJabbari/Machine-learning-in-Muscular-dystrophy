import random
import numpy as np


def discretize(data, bins_num):
#    sort_index = np.argsort(data)
    split = np.array_split(np.sort(data), bins_num)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
#    return discrete, sort_index, cutoffs
    return discrete, cutoffs


if __name__ == '__main__':
#    dat = np.arange(1,13)/2.0
    dat = [ 11, 0.5, 4, 2, 8,  1., 10 , 1.5,  2.,   2.5,  3.,   3.5,  4.,   4.5,  5.,   5.5,  6. ]    
    discrete_dat, cutoff = discretize(dat, 3)
#    discrete_dat, sort_index, cutoff = discretize(dat, 3)
    print "dat: {}".format(dat)
#    print "Index:", sort_index
    print "discrete_dat: {}".format(discrete_dat)
    print "cutoff: {}".format(cutoff)
