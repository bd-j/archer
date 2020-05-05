#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

flag_names = ["No_Gaia", "Fiber_130", "Nband_lt_5", "low_SNR_dup", "SNR_lt_1",
              "no_dist", "hot", "bad_spec_fit", "bad_phot_fit", "high_vrot",
              "bad_rv", "high_vtan"]
nflag = len(flag_names)

flags = [''.join(['1' if ii == j  else '0' for ii in range(nflag)])
         for j in range(nflag)]  
flags = [nflag * '0'] + flags

flagval = {}
for ii,ff in enumerate(flags):
    flagval[ii] = int(ff,2)


def flag_set(bitmask, flag_name, only=False):
    ind = flag_names.index(flag_name) + 1
    assert ind >= 0
    if only:
        f = bitmask == flagval[ind]
    else:
        f = np.bitwise_and(bitmask, flagval[ind]) > 0
    
    return f

def make_bitmask(flags):
    bitmask = 0
    for f in flags:
        ind = len(flag_names) - flag_names.index(f) - 1
        bitmask += (1 << ind)
    return bitmask
