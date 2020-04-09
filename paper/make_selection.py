#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def rcat_select(rcat, rcat_r, dly=0):

    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] >= 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 2.5 + dly)

    try:
        good_l = (rcat["Lz_err"] < 3e3) & (rcat["Ly_err"] < 3e3)
        good_l = good_l & (np.abs(rcat_r["lx"]) < 10) & (rcat["afe"] < (-0.3*rcat["FeH"]+0.2))
        good = good & good_l
    except:
        print("No L error cut")

    return good, sgr



def count_selections(rcat, rcat_r, dly=0):

    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] >= 3) &
            (rcat["logg"] < 3.5))
    sgr = rcat["Ly"]/1e3 < (-0.3 * rcat["Lz"]/1e3 - 2.5 + dly)
    good_chem = (rcat["FeH"] >= -3) & (rcat["afe"] < (-0.3*rcat["FeH"]+0.2))
    good_lerr = (rcat["Lz_err"] < 3e3) & (rcat["Ly_err"] < 3e3)
    good_lx = (np.abs(rcat_r["lx"]) < 10)

 
    n_tot = len(rcat)
    n_good = good.sum()
    n_good_sgr = (good & sgr).sum()
    n_good_chem_sgr = (good & good_chem & sgr).sum()
    n_good_lerr_sgr = (good & good_chem & good_lerr & sgr).sum()
    n_all_good_sgr = (good & good_chem & good_lerr & good_lx & sgr).sum()
    
