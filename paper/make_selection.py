#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def rcat_select(rcat, rcat_r, dly=0):

    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] > 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25 + dly)

    try:
        good_l = (rcat["Lz_err"] < 0.3e4) & (rcat["Ly_err"] < 0.3e4) & (np.abs(rcat_r["lx"]) < 1.0)
        good = good & good_l
    except:
        print("No L error cut")

    return good, sgr
