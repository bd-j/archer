#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from .cornerplot import get_cmap, twodhist


def samples_struct(starname, msID="V2.4", nsamples=1000):
    fmt = "samples/{starname}_MSG_{msID}.dat"
    fn = fmt.format(msID=msID, starname=starname)
    with open(fn, "r") as f:
        header = f.readline()
    cols = header.split()
    dt = np.dtype([(c, np.float) for c in cols])
    dat = np.genfromtxt(fn, skip_header=1, dtype=dt)
    p = np.exp(dat['logwt']-dat['logz'][-1])
    s = np.random.choice(len(dat), p=p/p.sum(), size=nsamples)
    return dat[s]


def get_Lstar_samples(samples, rcat_row, gc_frame=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rcat = rcat_row
    n = len(samples)

    # ra and dec are basically noiseless
    ra = np.ones(n) * rcat["RA"]
    dec = np.ones(n) * rcat["DEC"]
    # add pm uncertainties as gaussians
    pmra = np.random.normal(rcat["GAIADR2_PMRA"], 
                            rcat["GAIADR2_PMRA_ERROR"], 
                            size=n)
    pmdec = np.random.normal(rcat["GAIADR2_PMDEC"], 
                             rcat["GAIADR2_PMDEC_ERROR"], 
                             size=n)
    # use the MS samples for dist, vrad
    distance = samples["Dist"] / 1e3
    vlos = samples["Vrad"]

    L = compute_Lstar(ra, dec, distance, pmra, pmdec, vlos,
                      gc_frame=gc_frame)

    return L

def Lhist(L, ax=None, color="black",
          levels=np.array([1.0 - np.exp(-0.5 * 1**2)])):
    _, p2, p1 = L.T
    X, Y, H, V, clevels, _ = twodhist(p1, p2, levels=levels, smooth=0.05)
    
    # filled contour    
    contour_cmap = get_cmap(color, levels)
    ax.contourf(X, Y, H, clevels, antialiased=True, colors=contour_cmap)

    ax.contour(X, Y, H, V, colors=color)
    return ax
