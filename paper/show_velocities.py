#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.fitting import best_model, sample_posterior


if __name__ == "__main__":

    nsig = 2
    config = rectify_config(parser.parse_args())

    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
    
    # selection
    good = (rcat["FLAG"] == 0) & (rcat["SNR"] > 3) & (rcat["logg"] < 3.5)
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)

    # trailing
    tsel = good & sgr & (rcat_r["lambda"] < 175)
    tlam = rcat_r[tsel]["lambda"]
    tmu, tsig, _ = best_model("fits/h3_trailing_fit.h5", tlam)
    to = np.argsort(tlam)
    
    # leading
    lsel = good & sgr & (rcat_r["lambda"] > 175)
    llam = rcat_r[lsel]["lambda"]
    lmu, lsig, _ = best_model("fits/h3_leading_fit.h5", llam)
    lo = np.argsort(llam)

    # plot setup
    ncol = 1
    rcParams = plot_defaults(rcParams)
    figsize = (7, 3.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(ncol, 1, height_ratios=ncol * [10],
                  left=0.1, right=0.83, hspace=0.2,)
    gsc = GridSpec(ncol, 1, left=0.85, right=0.86, hspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    fcolor = "black"
    cmap = "magma"
    
    # plot fits
    ax.plot(tlam[to], tmu[to], color=fcolor, linestyle="--")
    ax.plot(llam[lo], lmu[lo], color=fcolor, linestyle="--")
    ax.fill_between(tlam[to], tmu[to] - nsig * tsig[to], tmu[to] + nsig * tsig[to],
                    alpha=0.3, color=fcolor, label="On-Arm")
    ax.fill_between(llam[lo], lmu[lo] - nsig * lsig[lo], lmu[lo] + nsig * lsig[lo],
                    alpha=0.3, color=fcolor)
    
    # plot data points
    sel = good & sgr
    cbh = ax.scatter(rcat_r[sel]["lambda"], rcat_r[sel]["vgsr"],
                     c=rcat[sel]["FeH"], vmin=-2.5, vmax=0.0, cmap=cmap,
                     s=9)
    
    ax.set_ylabel(r"V$_{\rm GSR}$ (km/s)")
    ax.set_xlabel(r"$\Lambda_{\rm Sgr}$")
    ax.set_ylim(-300, 300)
    
    cax = fig.add_subplot(gsc[0, 0])
    pl.colorbar(cbh, cax=cax, label=r"[Fe/H]")#, orientation="horizontal")

    pl.show()