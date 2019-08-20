#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Script to plot up a Sgr selection
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
    lmr = (np.random.uniform(size=len(lm)) < 0.3) & (~lmhsel)
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

    # make a superplot
    nsel = 2
    figsize = (12, 8)
    ms = 3
    fig, axes = pl.subplots(2, nsel, sharex="col", sharey="col",
                            figsize=figsize)

    # --- lsgr vs phi
    #pfig, paxes = pl.subplots(1, 2, sharey=True, sharex=True)
    paxes = axes[:, 0]
    ax = paxes[0]
    cb = ax.scatter(phisgr_lm[lmr][ro], lsgr_lm[lmr][ro], c=lm["Lmflag"][lmr][ro],
                    marker='+', alpha=0.3, vmin=-2, vmax=3, s=16 )
    cb = ax.scatter(phisgr_lm[lmhsel][ho], lsgr_lm[lmhsel][ho], c=lm["Lmflag"][lmhsel][ho],
                    marker='o', alpha=0.6, vmin=-2, vmax=3, s=16)

    p = lm["Pcol"] - lm["Pcol"].min() + 1.0
    p = None
    hist2d(phisgr_lm, lsgr_lm, ax=ax, span=[(0, 1), (-10000, 15000)], weights=p)

    ax = paxes[1]
    ax.plot(phisgr[good & ~sel], lsgr[good & ~sel], 'o', markersize=ms, alpha=0.3, color="grey")
    ax.scatter(phisgr[good & sel], lsgr[good & sel], c=feh[good & sel],
               marker='o', s=16, vmin=-2.5, vmax=0, alpha=0.7)
    
    # prettify
    ax.set_ylim(-2000, 15000)
    [ax.set_ylabel(r"$L_{Sgr}$") for ax in paxes]
    paxes[1].set_xlabel(r"$\cos \phi_{Sgr}$")# for ax in paxes]
    #[ax.axhline(-lslim, linestyle=":", color="tomato") for ax in paxes]
    [ax.axhline(lslim, linestyle=":", color="tomato") for ax in paxes]
    [ax.axvline(philim, linestyle=":", color="tomato") for ax in paxes]
    #pfig.colorbar(cb, ax=paxes)
    paxes[0].text(0.1, 0.9, "LM10", transform=paxes[0].transAxes)
    paxes[1].text(0.1, 0.9, "H3 Giants", transform=paxes[1].transAxes)

    # --- E-Lsgr ---
    #efig, eaxes = pl.subplots(1, 2, sharey=True, sharex=True)
    #eaxes = axes[1, :]
    #eax = eaxes[0]
    #ec = eax.scatter(lsgr_lm[lmr][ro], etot_lm[lmr][ro], c=lm["Lmflag"][lmr][ro],
    #                 marker='+', alpha=0.3, vmin=-2, vmax=3)
    #ec = eax.scatter(lsgr_lm[lmhsel][ho], etot_lm[lmhsel][ho], c=lm["Lmflag"][lmhsel][ho],
    #                 marker='o', alpha=0.3, vmin=-2, vmax=3, s=14,)
    #eax = eaxes[1]
    #eax.plot(lsgr[good & ~sel], etot[good & ~sel], 'o', markersize=ms, alpha=0.3, label="H3")
    #eax.plot(lsgr[good & sel], etot[good & sel], 'o', markersize=ms, alpha=0.3, label="H3 Selected")
    
    # Prettify
    #[a.set_ylim(-2e5, -5e4) for a in eaxes]
    #[a.set_xlim(-1e4, 2e4) for a in eaxes]
    #[a.set_xlabel(r"$L_{sgr}$") for a in eaxes]
    #eaxes[0].set_ylabel(r"$E_{tot}$")
    #[ax.axvline(lslim, linestyle=":", color="tomato") for ax in eaxes]
    #[ax.axhline(elim, linestyle=":", color="tomato") for ax in eaxes]


    # --- Lx - Ly ---
    #lfig, laxes = pl.subplots(1, 2, sharey=True, sharex=True)
    laxes = axes[:, 1]
    ax = laxes[0]
    lc = ax.scatter(lz_lm[lmr][ro], ly_lm[lmr][ro], c=lm["Lmflag"][lmr][ro],
                    marker="+", alpha=0.3, vmin=-2, vmax=3)
    lc = ax.scatter(lz_lm[lmhsel][ho], ly_lm[lmhsel][ho], c=lm["Lmflag"][lmhsel][ho],
                    marker="o", alpha=0.3, vmin=-2, vmax=3, s=14)
    ax = laxes[1]
    ax.plot(lz[good & ~sel], ly[good & ~sel], 'o', markersize=ms, alpha=0.3, color='grey')
    ax.scatter(lz[good & sel], ly[good & sel], c=feh[good & sel], 
               marker='o', s=16, vmin=-2.5, vmax=0.0, alpha=0.7)
    
    # prettify
    [ax.set_ylabel(r"$L_y$")  for ax in laxes]
    laxes[1].set_xlabel(r"$L_z$")
    laxes[0].set_ylim(-14000, 10000)
    laxes[0].set_xlim(-10000, 10000)
    [ax.plot(y, x, linestyle=":", color="tomato", linewidth=3) for ax in laxes]


    if savefigs:
        fig.tight_layout()
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/selection_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)


    sys.exit()

    # --- Vgsr lambda ----
    vfig, vaxes = pl.subplots(1, 2, sharey=True, sharex=True)
    ax = vaxes[0]
    vc = ax.scatter(lm["lambda"][lmr][ro], lm["V_gsr"][lmr][ro], c=lm["Lmflag"][lmr][ro],
                    marker="+", alpha=0.3, vmin=-2, vmax=3)
    vc = ax.scatter(lm["lambda"][lmhsel][ho], lm["V_gsr"][lmhsel][ho], c=lm["Lmflag"][lmhsel][ho],
                    marker="o", alpha=0.3, vmin=-2, vmax=3, s=14)
    ax = vaxes[1]
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["V_gsr"], 'o', alpha=0.3)
    ax.plot(rcat[good & sel]["Sgr_l"], rcat[good & sel]["V_gsr"], 'o', alpha=0.5)
    ax.set_ylim(-300, 300)
    vaxes[0].set_ylabel(r"$V_{GSR}$")
    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in vaxes]

    # --- polar sky position ---

    lfig = pl.figure()
    lax = pl.subplot(projection="polar")
    cb = lax.scatter(np.deg2rad(360 - lm["lambda"]), lm["dist"], c=phisgr_lm,
                marker='o', vmin=-1, vmax=1, alpha=0.4, s=2)
    lax.set_rmax(100)

    qfig, axes = pl.subplots(2, 1, squeeze=False, figsize=(8, 12.5), sharex=True)
    projections = ["xz", "xy"]
    ascale = 25
    from matplotlib import cm, colors
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=-1., vmax=1., clip=False)
    lcatz = [phisgr_lm, phisgr_lm]
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)

    sel = np.random.uniform(size=len(lm)) < 0.2
    cbl = [lm_quiver(lm[sel], z[sel], vtot=vtot_lm[sel], show=s, ax=ax, 
                     scale=ascale, cmap=cmap, norm=norm)
           for s, ax, z in zip(projections, axes[:, 0], lcatz)]

    pl.show()
    sys.exit()

