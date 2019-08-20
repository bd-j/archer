#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Script to plot some basic Lmabda, vgsr, dist plots for mocks and data with
different selections
"""

import sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
import gala

from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc

from cornerplot import _hist2d as hist2d

from utils import read_lm, read_segue
from utils import gc_frame_law10, sgr_law10, sgr_fritz18
from utils import get_values

rcParams["font.family"] = "serif"


if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = True
    segue_cat = False
    noisiness = "noisy"  # "noisy" | "noiseless"

    seguefile = "../data/catalogs/ksegue_gaia_v5.fits"
    rcat_vers = "1_4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)

    # --- L & M 2010 model ---
    lm = read_lm(lmockfile)

    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    data_name = "H3v{}".format(rcat_vers)
    if segue_cat:
        rcat = read_segue(seguefile, rcat.dtype)
        data_name = "KSEG"

    # --- Quantity shortcuts ---
    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr_law10)
    feh = rcat["FeH"]
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq

    # --- Basic selections ---
    # selections
    basic = ((rcat["FLAG"] == 0) & np.isfinite(rcat["Z_gal"]))
    giant = (rcat["logg"] < 3.5)
    extra = ((rcat["Vrot"] < 5) & (rcat["SNR"] > 3) &
             (rcat["BHB"] == 0) & (rcat["Teff"] < 7000) &
             (rcat["V_tan"] < 500))
    good = basic & giant & extra

    # Lm10 selections
    lmhsel = (lm["in_h3"] == 1)
    lmr = (np.random.uniform(size=len(lm)) < 0.1) & (~lmhsel)
    # for random order
    ho = np.random.choice(lmhsel.sum(), size=lmhsel.sum(), replace=False)
    ro = np.random.choice(lmr.sum(), size=lmr.sum(), replace=False)

    # Sgr selections
    # Ly - Lz
    x, y = [(-3500, -500), (4000, -6000)]
    m = np.diff(y) / np.diff(x)
    b = y[0] - m * x[0]
    m, b = m[0], b[0]
    lsel = lz < (m * ly + b)

    # phi - lsgr
    philim, lslim = 0.75, 1500
    phisel = (phisgr > philim) & (lsgr > lslim)
    retro = (phisgr < -0.5) & (lsgr < -5000)

    # etot -lsgr
    elim = -170000
    esel = (lsgr > lslim) & (etot < 0) & (etot > elim)

    # --- SET THE SELECTION ----
    #sel, selname = phisel, "phisel"
    #sel, selname = lsel, "LzLysel"
    #sel, selname = esel, "LsEsel"
    sel, selname = phisel & lsel & esel, "allsel"

    # Plot defaults
    ms = 3  # markersize
    # make a superplot
    ncol = 2
    figsize = (16, 10)
    fig, axes = pl.subplots(2, ncol, sharex="col", sharey="col",
                            figsize=figsize)

    # --- Vgsr vs Lambda ---
    vlaxes = axes[:, 0]
    ax = vlaxes[0]
    cb = ax.scatter(lm[lmr][ro]["lambda"], lm[lmr][ro]["V_gsr"], c=lm[lmr][ro]["Lmflag"],
                    marker='+', alpha=0.3, vmin=-2, vmax=3, s=16 )
    cb = ax.scatter(lm[lmhsel][ho]["lambda"], lm[lmhsel][ho]["V_gsr"], c=lm[lmhsel][ho]["Lmflag"],
                    marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, edgecolor="")

    ax = vlaxes[1]
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["V_gsr"], 
            'o', markersize=ms, alpha=0.3, color="grey", zorder=1)
    ax.scatter(rcat[good & sel]["Sgr_l"], rcat[good & sel]["V_gsr"], c=feh[good & sel], 
               marker='o', s=16, vmin=-2.5, vmax=0.0, alpha=0.6, zorder=2)

    # prettify
    ax.set_ylim(-300, 300)
    [ax.set_ylabel(r"$V_{GSR}$") for ax in vlaxes]
    vlaxes[1].set_xlabel(r"$\Lambda_{Sgr}$")

    # --- Rgc vs Lambda
    dlaxes = axes[:, 1]
    ax = dlaxes[0]
    cb = ax.scatter(lm[lmr][ro]["lambda"], lm[lmr][ro]["dist"], c=lm[lmr][ro]["Lmflag"],
                    marker='+', alpha=0.3, vmin=-2, vmax=3, s=16 )
    cb = ax.scatter(lm[lmhsel][ho]["lambda"], lm[lmhsel][ho]["dist"], c=lm[lmhsel][ho]["Lmflag"],
                    marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, edgecolor="")

    ax = dlaxes[1]
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["dist_adpt"], 
            'o', markersize=ms, alpha=0.3, color="grey", zorder=1)
    ax.scatter(rcat[good & sel]["Sgr_l"], rcat[good & sel]["dist_adpt"], c=feh[good & sel],
               marker='o', s=16, vmin=-2.5, vmax=0.0, alpha=0.6, zorder=2)

    # prettify
    ax.set_ylim(0, 80)
    [ax.set_ylabel(r"$D_{Sun}$") for ax in dlaxes]
    dlaxes[1].set_xlabel(r"$\Lambda_{Sgr}$")

    if savefigs:
        fig.tight_layout()
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/vsLambda_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)
    else:
        pl.show()

    # --- Vgsr vs Dist ---
    #dvaxes = axes[:, 2]
    #ax = dvaxes[0]
    #cb = ax.scatter(lm[lmr][ro]["dist"], lm[lmr][ro]["V_gsr"], c=lm[lmr][ro]["Lmflag"],
    #                marker='+', alpha=0.3, vmin=-2, vmax=3, s=16 )
    #cb = ax.scatter(lm[lmhsel][ho]["dist"], lm[lmhsel][ho]["V_gsr"], c=lm[lmhsel][ho]["Lmflag"],
    #                marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, edgecolor="")

    #ax = dvaxes[1]
    #ax.plot(rcat[good & ~sel]["dist_adpt"], rcat[good & ~sel]["V_gsr"], 'o', markersize=ms, alpha=0.3)
    #ax.plot(rcat[good & sel]["dist_adpt"], rcat[good & sel]["V_gsr"], 'o', markersize=ms, alpha=0.3)

    # prettify
    #ax.set_ylim(-300, 300)
    #ax.set_xlim(0, 80)
    #[ax.set_ylabel(r"$V_{GSR}$") for ax in dvaxes]
    #dvaxes[1].set_xlabel(r"$D_{Sun}$")

    #pl.show()
    #sys.exit()
