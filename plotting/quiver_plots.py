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
pl.ion()
from astropy.io import fits

from utils import hquiver
from utils import read_lm, read_segue
from utils import gc_frame_law10, sgr_law10, sgr_fritz18
from utils import get_values

from matplotlib.pyplot import rcParams
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
    feh = np.clip(rcat["FeH"], -2.5, 0.0)
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)

    # --- Basic selections ---
    # selections
    basic = ((rcat["FLAG"] == 0) & np.isfinite(rcat["Z_gal"]))
    giant = (rcat["logg"] < 3.5)
    extra = ((rcat["Vrot"] < 5) & (rcat["SNR"] > 3) &
             (rcat["BHB"] == 0) & (rcat["Teff"] < 7000) &
             (rcat["V_tan"] < 600))
    good = basic & giant & extra

    # Lm10 selections
    lmhsel = (lm["in_h3"] == 1)
    lmr = (np.random.uniform(size=len(lm)) < 0.1) & (~lmhsel)
    lmsel = lmr | lmhsel
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
    gs = GridSpec(2, 2, width_ratios= [10, 10], left=0.1, right=0.87, wspace=0.2)
    gsc = GridSpec(2, 1, left=0.9, right=0.93)

    laxes = [fig.add_subplot(gs[0, 0])]
    laxes.append(fig.add_subplot(gs[0, 1]))
    haxes = [fig.add_subplot(gs[1, 0], sharex=laxes[0], sharey=laxes[0])]
    haxes.append(fig.add_subplot(gs[1, 1], sharex=laxes[1], sharey=laxes[1]))
    
    axes = np.vstack([laxes, haxes])
    
    #cbl = [lm_quiver(lm[lmsel], z[lmsel], vtot=vtot_lm[lmsel], show=s, ax=ax, scale=ascale)
    #       for s, ax, z in zip(projections, axes[0, :], lcatz)]
    cbl = [hquiver(lm[lmr], z[lmr], vtot=vtot_lm[lmr], show=s, ax=ax, scale=ascale, alpha=0.3)
           for s, ax, z in zip(projections, laxes, lcatz)]
    cbl = [hquiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, laxes, lcatz)]
    #cblh = [lm_quiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel], show=s, ax=ax, scale=ascale)
    #        for s, ax, z in zip(projections, axes[:, 1], lcatz)]
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

    [ax.xaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[0, :]]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[:, 1:].flat]

    axes[0, 0].text(0.1, 0.9, "LM10", transform=axes[0,0].transAxes, fontsize=14)
    axes[1, 0].text(0.1, 0.9, "H3 Giants", transform=axes[1,0].transAxes, fontsize=14)
    for i in range(ncol-1):
        xl, yl = projections[i]
        for j in range(nrow):
            axes[j, i].set_xlabel(r"{}$_{{\rm GC}}$ (kpc)".format(xl.upper()))
            axes[j, i].set_ylabel(r"{}$_{{\rm GC}}$ (kpc)".format(yl.upper()))

    # colorbars
    cax = fig.add_subplot(gsc[0,-1])
    pl.colorbar(cbl[-1], cax=cax, label=r"Arm #")
    cax2 = fig.add_subplot(gsc[1,-1])
    pl.colorbar(cbh[-1], cax=cax2, label=r"[Fe/H]")

 

    fig.savefig("figures/quiver_placeholder.png", dpi=300)
    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())
    
    pl.show()

