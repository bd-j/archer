#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from .cornerplot import get_cmap, twodhist
from .quantities import compute_lstar
from .catalogs import required_columns, derived_columns


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

    # make homogenized catalog of the samples
    newcols = [r[0] for r in required_columns + derived_columns]
    dtype = np.dtype([(c, np.float32) for c in newcols])
    scat = np.zeros(n, dtype=dtype)

    # ra and dec are basically noiseless
    scat["ra"] = np.ones(n) * rcat["RA"]
    scat["dec"] = np.ones(n) * rcat["DEC"]
    # add pm uncertainties as gaussians
    scat["pmra"] = np.random.normal(rcat["GAIADR2_PMRA"], 
                                    rcat["GAIADR2_PMRA_ERROR"], 
                                    size=n)
    scat["pmdec"] = np.random.normal(rcat["GAIADR2_PMDEC"], 
                                     rcat["GAIADR2_PMDEC_ERROR"], 
                                     size=n)
    # use the minesweeper samples for dist, vrad
    scat["dist"] = samples["Dist"] / 1e3
    scat["vrad"] = samples["Vrad"]

    L = compute_lstar(scat, gc_frame=gc_frame)

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
