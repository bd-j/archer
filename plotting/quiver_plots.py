#!/usr/bin/python

"""Script to plot velocity vectors for H3 stars (and Sgr mocks) in X-Z, Y-Z
projections.

Also plots:
 * Vtheta-Vr
 * v_gsr vs \Lambda_sgr
 * B_sgr vs \Lambda_sgr
"""

import sys
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

from utils import hquiver
from utils import read_lm, read_segue
from utils import gc_frame_law10, sgr_law10, sgr_fritz18
from utils import get_values
from outlier import get_rcat_selections, delta_v, vel_outliers

from matplotlib.pyplot import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'

pl.ion()

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
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)

    good = basic & extra & giant
    sel, selname = lsel, "lsel"
    chiv = delta_v(rcat)
    intail, inlead, outtail, outlead = vel_outliers(rcat, good & sel)

    lead = rcat["Sgr_l"] < 150
    trail = rcat["Sgr_l"] > 200

    # --- L & M 2010 model ---
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)
    lm = read_lm(lmockfile)
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)

    # Lm10 selections
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(size=len(lm)) < 1.0)
    lmr = (np.random.uniform(size=len(lm)) < 1.0) & (~lmhsel)
    # for random order
    ho = np.random.choice(lmhsel.sum(), size=good.sum(), replace=True)
    if segue_cat:
        n = good.sum()
    else:
        n = int(len(lm) * 1.0 / (lmhsel.sum() * 1.0) * (good & lsel).sum())
    ro = np.random.choice(lmr.sum(), size=n, replace=True)

    # --- Quiver ---
    nrow, ncol = 2, 3
    projections = ["xz", "xy"]
    ascale = 25
    hcmap = "magma"

    lcatz = [lm["Lmflag"], lm["Lmflag"]]
    #lcatz = [lmf, lmf]
    #lcatz = [lm["v"]/vtot_lm, lm["w"]/vtot_lm]
    rcatz = [feh, feh]

    bb = good

    figsize = 10, 8
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, width_ratios=[10, 10],
                  left=0.1, right=0.87, wspace=0.2)
    gsc = GridSpec(2, 1, left=0.9, right=0.93)

    laxes = [fig.add_subplot(gs[0, 0])]
    laxes.append(fig.add_subplot(gs[0, 1]))
    haxes = [fig.add_subplot(gs[1, 0], sharex=laxes[0], sharey=laxes[0])]
    haxes.append(fig.add_subplot(gs[1, 1], sharex=laxes[1], sharey=laxes[1]))

    axes = np.vstack([laxes, haxes])

    cbl = [hquiver(lm[lmr][ro], z[lmr][ro], vtot=vtot_lm[lmr][ro],
                   show=s, ax=ax, scale=ascale, alpha=0.3)
           for s, ax, z in zip(projections, laxes, lcatz)]
    if not segue_cat:
        cbl = [hquiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel],
                       show=s, ax=ax, scale=ascale)
               for s, ax, z in zip(projections, laxes, lcatz)]
    cbh = [hquiver(rcat[good], None, vtot=vtot[good],
                   show=s, ax=ax, scale=ascale, alpha=0.3, color="grey")
           for s, ax, z in zip(projections, haxes, rcatz)]
    cbh = [hquiver(rcat[bb & sel], z[bb & sel], vtot=vtot[bb & sel],
                   show=s, ax=ax, scale=ascale, cmap=hcmap)
           for s, ax, z in zip(projections, haxes, rcatz)]

    axes[0, 0].set_xlim(-70, 40)  # x in x-z
    axes[0, 0].set_ylim(-47, 67)  # z in x-z
    axes[0, 1].set_xlim(-70, 40)  # x in x-y
    axes[0, 1].set_ylim(-32, 42)  # y in x-y

    [ax.xaxis.set_tick_params(which='both', labelbottom=True)
     for ax in axes[0, :]]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True)
     for ax in axes[:, 1:].flat]

    if segue_cat:
        dlabel = "Segue"
    else:
        dlabel = "H3"
    axes[0, 0].text(0.1, 0.9, "LM10", transform=axes[0, 0].transAxes, fontsize=14)
    axes[1, 0].text(0.1, 0.9, "{} Giants".format(dlabel), transform=axes[1, 0].transAxes, fontsize=14)
    for i in range(ncol - 1):
        xl, yl = projections[i]
        for j in range(nrow):
            axes[j, i].set_xlabel(r"{}$_{{\rm GC}}$ (kpc)".format(xl.upper()))
            axes[j, i].set_ylabel(r"{}$_{{\rm GC}}$ (kpc)".format(yl.upper()))

    # colorbars
    cax = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cbl[-1], cax=cax, label=r"Arm #")
    cax2 = fig.add_subplot(gsc[1, -1])
    pl.colorbar(cbh[-1], cax=cax2, label=r"[Fe/H]")

    names = data_name, noisiness, selname, ext
    fig.savefig("figures/quiver_{}_{}_{}.{}".format(*names), dpi=300)
    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())

    pl.show()
