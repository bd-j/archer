#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def rcat_select(rcat, rcat_r, dly=0.0, achilles_file=None):

    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] >= 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 2.5 + dly)

    try:
        good_l = (rcat["Lz_err"] < 3e3) & (rcat["Ly_err"] < 3e3)
        good_l = good_l & (np.abs(rcat_r["lx"]) < 10) & (rcat["afe"] < (-0.3*rcat["FeH"]+0.2))
        good = good & good_l
    except:
        print("No L error cut")

    if achilles_file:
        raise(NotImplementedError)
        from astropy.io import fits
        dat = fits.getdata(achilles_file)
        ach = np.where(rcat["starname"] == dat["starname"])

    #return np.ones(len(rcat), dtype=bool), np.ones(len(rcat), dtype=bool) 
    return good, sgr


def extra_sgr(rcat, rcat_r):
    """achilles_mask = ((rcat['Lz']/rcat['circ'])>-0.6) & ((rcat['Lz']/rcat['circ'])<-0.25) \
    & (~rcat['lsel']) & (rcat['E_tot_pot1']>-1.2) & (rcat['Lz']<-0.75) \
    & (elz_mask) & ~hs_flag & ~wukong_mask_final & (rcat['GAIADR2_PMRA']<0.5)  & (rcat['GAIADR2_PMRA']>-3)
    """
    pass


def gc_select(gcat):
    sgr_gcs = {'Whiting 1': -0.7, 'Pal 12': -0.85,
               'NGC 6715': -1.49, 'Ter 7': -0.32, 'Arp 2': -1.75, 'Ter 8': -2.16, 
               'NGC 2419': -2.15,}# "NGC 5466": -1.99}# "NGC5634": -1.88, "Pal 2": -1.42} #'NGC 5824': -1.91, 'NGC 4147': -1.80}
    inds = np.array([g["Name"] in sgr_gcs.keys() for g in gcat], dtype=bool)
    feh = np.array([sgr_gcs.get(g["Name"], -2) for g in gcat])
    return inds, feh


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
    
    goodish = ((rcat["SNR"] >= 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 2.5 + dly)
