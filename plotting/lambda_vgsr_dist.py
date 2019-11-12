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
    np.random.seed(101)

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
    sel, selname = lsel, "lsel"
    chiv = delta_v(rcat)
    intail, inlead, outtail, outlead = vel_outliers(rcat, good & sel)

    trail = rcat["Sgr_l"] < 150
    lead = rcat["Sgr_l"] > 200

    # --- L & M 2010 model ---
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)
    lm = read_lm(lmockfile)
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq

    # Lm10 selections
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(size=len(lm)) < 1.0)
    lmr = (np.random.uniform(size=len(lm)) < 1.0) & (~lmhsel)
    # for random order
    ho = np.random.choice(lmhsel.sum(), size=good.sum(), replace=True)
    if segue_cat:
        n = good.sum()
    else:
        n = int(len(lm) * 1.0 / (lmhsel.sum() * 1.0) * (lsel & good).sum())
    ro = np.random.choice(lmr.sum(), size=n, replace=True)


    # --- Plot Defaults and Setup ---
    ms = 2  # markersize
    hcmap = "magma"
    # make a superplot
    ncol = 3
    figsize = (14, 8)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, ncol, width_ratios=ncol * [10],
                  left=0.1, right=0.87, wspace=0.2)
    gsc = GridSpec(2, 1, left=0.9, right=0.93)

    # --- Vgsr vs Lambda ---
    vlaxes = [fig.add_subplot(gs[0, 0])]
    ax = vlaxes[0]
    cb = ax.scatter(lm[lmr][ro]["lambda"], lm[lmr][ro]["V_gsr"],
                    c=lm[lmr][ro]["Lmflag"],
                    marker='+', alpha=0.1, vmin=-2, vmax=3, s=16)
    if not segue_cat:
        cb = ax.scatter(lm[lmhsel][ho]["lambda"], lm[lmhsel][ho]["V_gsr"],
                        c=lm[lmhsel][ho]["Lmflag"],
                        marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, linewidth=0)

    ax = fig.add_subplot(gs[1, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["V_gsr"],
            'o', markersize=ms, alpha=0.2, color="grey", zorder=1, mew=0)
    cbh = ax.scatter(rcat[good & sel]["Sgr_l"], rcat[good & sel]["V_gsr"],
                     c=feh[good & sel], vmin=-2.5, vmax=0.0, cmap=hcmap,
                     marker='o', s=16, alpha=0.6, zorder=2, linewidth=0)

    # prettify
    ax.set_ylim(-300, 300)
    [ax.set_ylabel(r"V$_{\rm GSR}$ (km/s)") for ax in vlaxes]
    vlaxes[1].set_xlabel(r"$\Lambda_{\rm Sgr}$")

    # --- Rgc vs Lambda
    dlaxes = [fig.add_subplot(gs[0, 1])]
    ax = dlaxes[0]
    cb = ax.scatter(lm[lmr][ro]["lambda"], lm[lmr][ro]["dist"],
                    c=lm[lmr][ro]["Lmflag"],
                    marker='+', alpha=0.1, vmin=-2, vmax=3, s=16)
    if not segue_cat:
        cb = ax.scatter(lm[lmhsel][ho]["lambda"], lm[lmhsel][ho]["dist"],
                        c=lm[lmhsel][ho]["Lmflag"],
                        marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, linewidth=0)

    ax = fig.add_subplot(gs[1, 1], sharey=dlaxes[0], sharex=dlaxes[0])
    dlaxes.append(ax)
    ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["dist_adpt"],
            'o', markersize=ms, alpha=0.2, color="grey", zorder=1, mew=0)
    cbh = ax.scatter(rcat[good & sel]["Sgr_l"], rcat[good & sel]["dist_adpt"],
                     c=feh[good & sel], vmin=-2.5, vmax=0.0, cmap=hcmap,
                     marker='o', s=16, alpha=0.6, zorder=2, linewidth=0)

    # prettify
    ax.set_ylim(0, 80)
    [ax.set_ylabel(r"D$_{\odot}$ (kpc)", labelpad=2) for ax in dlaxes]
    dlaxes[1].set_xlabel(r"$\Lambda_{\rm Sgr}$")

    # colorbars
    cax = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cb, cax=cax, label=r"Arm #")
    cax2 = fig.add_subplot(gsc[1, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")

    if savefigs:
        #fig.tight_layout()
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/vsLambda_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)
    else:
        pl.show()

    # --- B vs Lambda
    if True:
        dlaxes = [fig.add_subplot(gs[0, 2])]
        ax = dlaxes[0]
        cb = ax.scatter(lm[lmr][ro]["lambda"], lm[lmr][ro]["beta"],
                        c=lm[lmr][ro]["Lmflag"],
                        marker='+', alpha=0.1, vmin=-2, vmax=3, s=16)
        if not segue_cat:
            cb = ax.scatter(lm[lmhsel][ho]["lambda"], lm[lmhsel][ho]["beta"],
                            c=lm[lmhsel][ho]["Lmflag"],
                            marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, linewidth=0)

        ax = fig.add_subplot(gs[1, 2], sharey=dlaxes[0], sharex=dlaxes[0])
        dlaxes.append(ax)
        ax.plot(rcat[good & ~sel]["Sgr_l"], rcat[good & ~sel]["Sgr_b"],
                'o', markersize=ms, alpha=0.2, color="grey", zorder=1, mew=0)
        cbh = ax.scatter(rcat[good & sel]["Sgr_l"], rcat[good & sel]["Sgr_b"],
                        c=feh[good & sel], vmin=-2.5, vmax=0.0, cmap=hcmap,
                        marker='o', s=16, alpha=0.6, zorder=2, linewidth=0)

        # prettify
        ax.set_ylim(-50, 30)
        [ax.set_ylabel(r"$B_{\rm Sgr}$", labelpad=2) for ax in dlaxes]
        dlaxes[1].set_xlabel(r"$\Lambda_{\rm Sgr}$")


    # --- Vgsr vs Dist ---
    if False:
        dvaxes = [fig.add_subplot(gs[0, 2])]
        ax = dvaxes[0]
        cb = ax.scatter(lm[lmr][ro]["dist"], lm[lmr][ro]["vgsr"],
                        c=lm[lmr][ro]["Lmflag"],
                        marker='+', alpha=0.1, vmin=-2, vmax=3, s=16)
        if not segue_cat:
            cb = ax.scatter(lm[lmhsel][ho]["dist"], lm[lmhsel][ho]["vgsr"],
                            c=lm[lmhsel][ho]["Lmflag"],
                            marker='o', alpha=0.5, vmin=-2, vmax=3, s=12, linewidth=0)

        ax = fig.add_subplot(gs[1, 2], sharey=dvaxes[0], sharex=dvaxes[0])
        dvaxes.append(ax)
        ax.plot(rcat[good & ~sel]["dist_adpt"], rcat[good & ~sel]["V_gsr"],
                    'o', markersize=ms, alpha=0.2, color="grey", zorder=1, mew=0)
        cbh = ax.scatter(rcat[good & sel]["dist_adpt"], rcat[good & sel]["V_gsr"],
                            c=feh[good & sel], vmin=-2.5, vmax=0.0, cmap=hcmap,
                            marker='o', s=16, alpha=0.6, zorder=2, linewidth=0)

        # prettify
        ax.set_ylim(-300, 300)
        ax.set_xlim(0, 80)
        [ax.set_ylabel(r"$V_{GSR}$") for ax in dvaxes]
        dvaxes[1].set_xlabel(r"$D_{Sun}$")

    #pl.show()
    #sys.exit()
    
        # colorbars
    cax = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cb, cax=cax, label=r"Arm #")
    cax2 = fig.add_subplot(gsc[1, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")

    if savefigs:
        #fig.tight_layout()
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/vsLambda_B_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)
    else:
        pl.show()


