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
from outlier import get_rcat_selections, delta_v, vel_outliers

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'

if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = True
    segue_cat = False
    noisiness = "noisy"  # "noisy" | "noiseless"
    rcat_vers = "1_4"

    if segue_cat:
        data_name = "KSEG"
    else:
        data_name = "H3v{}".format(rcat_vers)

    # --- H3 cat and selections ----
    philim, lslim = 0.75, 1500
    x, y = [(-3500, -500), (4000, -6000)]
    blob = get_rcat_selections(segue_cat=segue_cat,
                               rcat_vers=rcat_vers,
                               sgr=sgr_law10,
                               philim=philim, lslim=lslim,
                               lcut=[x, y])
    rcat, basic, giant, extra, lsel, phisel, esel = blob

    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr_law10)
    feh = np.clip(rcat["FeH"], -2.5, 0.0)

    good = basic & extra & giant
    sel, selname = lsel & phisel & esel, "allsel"
    chiv = delta_v(rcat)
    intail, inlead, outtail, outlead = vel_outliers(rcat, good & sel)
    out = outtail | outlead

    lead = rcat["Sgr_l"] < 150
    trail = rcat["Sgr_l"] > 200

    # --- L & M 2010 model ---
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)
    lm = read_lm(lmockfile)
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq

    # Lm10 selections
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(size=len(lm)) < 0.5)
    lmr = (np.random.uniform(size=len(lm)) < 0.1) & (~lmhsel)
    # for random order
    ho = np.random.choice(lmhsel.sum(), size=lmhsel.sum(), replace=False)
    ro = np.random.choice(lmr.sum(), size=lmr.sum(), replace=False)

    # --- make a superplot ---
    nsel = 2
    figsize = (10, 8)
    ms = 2
    hcmap = "magma"

    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, width_ratios=[10, 10],
                  left=0.1, right=0.87, wspace=0.28)
    gsc = GridSpec(2, 1, left=0.9, right=0.93)

    zorder = np.argsort(feh[good & sel])[::-1]

    # -----------------------
    # --- L_sgr vs Phi ---
    paxes = [fig.add_subplot(gs[0, 0])]
    ax = paxes[0]
    cb = ax.scatter(phisgr_lm[lmr][ro], lsgr_lm[lmr][ro],
                    c=lm["Lmflag"][lmr][ro], vmin=-2, vmax=3,
                    marker='+', alpha=0.2, s=16)
    cb = ax.scatter(phisgr_lm[lmhsel][ho], lsgr_lm[lmhsel][ho],
                    c=lm["Lmflag"][lmhsel][ho], vmin=-2, vmax=3,
                    marker='*', alpha=0.5, s=16, linewidth=0)

    p = lm["Pcol"] - lm["Pcol"].min() + 1.0
    p = None
    hist2d(phisgr_lm, lsgr_lm, ax=ax,
           span=[(0, 1), (-10000, 15000)], weights=p)

    paxes.append(fig.add_subplot(gs[1, 0], sharey=paxes[0], sharex=paxes[0]))
    ax = paxes[1]
    ax.plot(phisgr[good & ~sel], lsgr[good & ~sel], 'o', markersize=ms, mew=0,
            alpha=0.3, color="grey")
    cbh = ax.scatter(phisgr[good & sel][zorder], lsgr[good & sel][zorder],
                     c=feh[good & sel][zorder], vmin=-2.5, vmax=0,
                     marker='o', s=16, alpha=0.6, cmap=hcmap, linewidth=0)
    _ = ax.scatter(phisgr[good & sel & out], lsgr[good & sel & out],
                     c=feh[good & sel & out], vmin=-2.5, vmax=0,
                     marker='*', s=25, alpha=1.0, linewidth=0, cmap=hcmap)

    # prettify
    ax.set_ylim(-1000, 15000)
    [ax.set_ylabel(r"L$_{\rm Sgr}$") for ax in paxes]
    paxes[1].set_xlabel(r"$\cos \, \phi_{\rm Sgr}$")  # for ax in paxes]
    [ax.axhline(lslim, linestyle=":", color="tomato") for ax in paxes]
    [ax.axvline(philim, linestyle=":", color="tomato") for ax in paxes]
    paxes[0].text(0.1, 0.9, r"LM10", transform=paxes[0].transAxes, fontsize=14)
    paxes[1].text(0.1, 0.9, r"H3 Giants", transform=paxes[1].transAxes, fontsize=14)

    # -------------------
    # --- Lx - Ly ---
    laxes = [fig.add_subplot(gs[0, 1])]
    ax = laxes[0]
    lc = ax.scatter(lz_lm[lmr][ro], ly_lm[lmr][ro], c=lm["Lmflag"][lmr][ro],
                    marker="+", alpha=0.2, vmin=-2, vmax=3)
    lc = ax.scatter(lz_lm[lmhsel][ho], ly_lm[lmhsel][ho],
                    c=lm["Lmflag"][lmhsel][ho], vmin=-2, vmax=3,
                    marker="o", alpha=0.5, s=14, linewidth=0)
    laxes.append(fig.add_subplot(gs[1, 1], sharey=laxes[0], sharex=laxes[0]))
    ax = laxes[1]
    ax.plot(lz[good & ~sel], ly[good & ~sel], 'o', markersize=ms, mew=0,
            alpha=0.3, color='grey')
    cbh = ax.scatter(lz[good & sel][zorder], ly[good & sel][zorder],
                     c=feh[good & sel][zorder], vmin=-2.5, vmax=0.0,
                     marker='o', s=16, alpha=0.6, cmap=hcmap, linewidth=0)

    # prettify
    [ax.set_ylabel(r"L$_{\rm y}$") for ax in laxes]
    laxes[1].set_xlabel(r"L$_{\rm z}$")
    laxes[0].set_ylim(-14000, 10000)
    laxes[0].set_xlim(-10000, 10000)
    [ax.plot(y, x, linestyle=":", color="tomato", linewidth=3) for ax in laxes]

    # colorbars
    cax = fig.add_subplot(gsc[0, -1])
    pl.colorbar(lc, cax=cax, label=r"Arm #")
    cax2 = fig.add_subplot(gsc[1, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")

    if savefigs:
        #fig.tight_layout()
        #fig.subplots_adjust(hspace=0.15)
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/selection_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)
    else:
        pl.show()

    # Misc plots
    sys.exit()

    # --- Vgsr lambda ----
    vfig, vaxes = pl.subplots(1, 2, sharey=True, sharex=True)
    ax = vaxes[0]
    vc = ax.scatter(lm["lambda"][lmr][ro], lm["V_gsr"][lmr][ro],
                    c=lm["Lmflag"][lmr][ro], vmin=-2, vmax=3,
                    marker="+", alpha=0.3)
    vc = ax.scatter(lm["lambda"][lmhsel][ho], lm["V_gsr"][lmhsel][ho],
                    c=lm["Lmflag"][lmhsel][ho], vmin=-2, vmax=3,
                    marker="o", alpha=0.3, s=14)
    ax = vaxes[1]
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["V_gsr"],
            marker='o', linestyle="", alpha=0.3)
    ax.plot(rcat[good & sel]["Sgr_l"], rcat[good & sel]["V_gsr"],
            marker='o', linestyle="", alpha=0.5)
    ax.set_ylim(-300, 300)
    vaxes[0].set_ylabel("$V_{GSR}$")
    [ax.set_xlabel("$\Lambda_{Sgr}$") for ax in vaxes]

    # --- polar sky position ---

    lfig = pl.figure()
    lax = pl.subplot(projection="polar")
    cb = lax.scatter(np.deg2rad(360 - lm["lambda"]), lm["dist"], c=phisgr_lm,
                     marker='o', vmin=-1, vmax=1, alpha=0.4, s=2)
    lax.set_rmax(100)

    qfig, axes = pl.subplots(2, 1, figsize=(8, 12.5), sharex=True)
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

    # --- E-Lsgr ---
    #efig, eaxes = pl.subplots(1, 2, sharey=True, sharex=True)
    #eaxes = axes[1, :]
    #eax = eaxes[0]
    #ec = eax.scatter(lsgr_lm[lmr][ro], etot_lm[lmr][ro],
    #                 c=lm["Lmflag"][lmr][ro],
    #                 marker='+', alpha=0.3, vmin=-2, vmax=3)
    #ec = eax.scatter(lsgr_lm[lmhsel][ho], etot_lm[lmhsel][ho],
    #                 c=lm["Lmflag"][lmhsel][ho],
    #                 marker='o', alpha=0.3, vmin=-2, vmax=3, s=14,)
    #eax = eaxes[1]
    #eax.plot(lsgr[good & ~sel], etot[good & ~sel], 'o',
    #         markersize=ms, alpha=0.3, label="H3")
    #eax.plot(lsgr[good & sel], etot[good & sel], 'o',
    #         markersize=ms, alpha=0.3, label="H3 Selected")

    # Prettify
    #[a.set_ylim(-2e5, -5e4) for a in eaxes]
    #[a.set_xlim(-1e4, 2e4) for a in eaxes]
    #[a.set_xlabel(r"$L_{sgr}$") for a in eaxes]
    #eaxes[0].set_ylabel(r"$E_{tot}$")
    #[ax.axvline(lslim, linestyle=":", color="tomato") for ax in eaxes]
    #[ax.axhline(elim, linestyle=":", color="tomato") for ax in eaxes]
