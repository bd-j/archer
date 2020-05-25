#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from archer.flags import flag_names, make_bitmask


def good_select(rcat, allow_flags=["hot", "high_vtan"], extras=True,
                snr_limit=3, max_rank=3):

    # allow some bits to be set
    assert np.all([f in flag_names for f in allow_flags])
    bitmask = np.int64(make_bitmask(allow_flags))
    ok_flags = np.bitwise_xor(np.bitwise_or(rcat["FLAG_BITMASK"], bitmask), bitmask) == 0

    # main quality selection
    with np.errstate(invalid="ignore"):
        good = (ok_flags & (rcat["SNR"] >= snr_limit) & (rcat["V_tan"] < 900) &
                (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3) &
                (rcat["XFIT_RANK"] <= (max_rank)))

    # remove large L uncertainties
    # remove weird chemistry
    if extras:
        with np.errstate(invalid="ignore"):
            good_l = (rcat["Lz_err"] < 3e3) & (rcat["Ly_err"] < 3e3)
            good_c = (rcat["afe"] < (-0.3*rcat["FeH"]+0.2))
            good = good & good_l & good_c
    else:
        print("No L error or chem cut")

    #return np.ones(len(rcat), dtype=bool), np.ones(len(rcat), dtype=bool)
    return good


def rcat_select(rcat, rcat_r, dly=0.0, flx=0.9, max_rank=3,
                allow_flags=["hot", "high_vtan"]):

    with np.errstate(invalid="ignore"):
        sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 2.5 + dly)
        sgr = sgr & (np.abs(rcat_r["lx"]) < flx*np.hypot(rcat_r["ly"], rcat_r["lz"]))
        good = good_select(rcat, allow_flags=allow_flags, max_rank=max_rank)
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


def count_selections(rcat, rcat_r, dly=0, flx=0.9, max_rank=2):

    goodish = good_select(rcat, extras=False, max_rank=max_rank)
    good_lerr = (rcat["Lz_err"] < 3e3) & (rcat["Ly_err"] < 3e3)
    good_chem = (rcat["afe"] < (-0.3*rcat["FeH"]+0.2))
    good_lx = (np.abs(rcat_r["lx"]) < flx*np.hypot(rcat_r["ly"], rcat_r["lz"]))

    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 2.5 + dly)


    n_tot = len(rcat)
    n_good = goodish.sum()
    n_good_lerr = (goodish & good_lerr).sum()
    n_sgr = (goodish & good_lerr & sgr).sum()
    n_good_chem_sgr = (goodish & good_lerr & sgr & good_chem).sum()
    n_good_lx_sgr = (goodish & good_lerr & sgr & good_lx).sum()
    n_all_good_sgr = (goodish & good_chem & good_lerr & good_lx & sgr).sum()
