#!/usr/bin/python

"""Script to examine metallicity distributions in Sgr
"""
import sys

import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from utils import get_values, sgr_law10
from utils import read_lm, read_segue
from plot_vdisp import read_from_h5
from outlier import get_rcat_selections, delta_v, vel_outliers

pl.ion()

if __name__ == "__main__":

    ext = "png"
    segue_cat = False
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
    feh = rcat["FeH"]

    good = basic & extra & giant
    sel, selname = lsel & phisel & esel, "allsel"
    chiv = delta_v(rcat)
    intail, inlead, outtail, outlead = vel_outliers(rcat, good & sel)

    trail = rcat["Sgr_l"] < 150
    lead = rcat["Sgr_l"] > 200
    zrange = (-2.8, 0.0)


    # --- FeH vs Lambda ---
    fig, ax = pl.subplots()
    arms = good & sel & (lead | trail)
    cb = ax.scatter(rcat[arms]["Sgr_l"], rcat[arms]["FeH"],
                    c=rcat[arms]["V_gsr"], alpha=0.8, vmin=-200, vmax=25)
    fig.colorbar(cb)
    pl.close(fig)

    # --- FeH vs cosphi ---
    plim = 0.8

    figsize = (15, 6)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, 
                  #height_ratios=[0.2, 1.0], 
                  width_ratios=[0.2, 3, 1, 1],
                  left=0.05, right=0.95, wspace=0.25)
    n=0
    axes = [fig.add_subplot(gs[i]) for i in range(n, n+4)]

    arms = good & esel & lsel & (lead | trail)
    #a.hist(phisgr[good & sel & out], range=(0.7, 1), bins=20, density=True, alpha=0.4) 
    ax = axes[1]
    cb = ax.scatter(phisgr[arms], rcat[arms]["FeH"], c=np.abs(chiv[arms]),
                    alpha=0.8, vmin=0, vmax=5, cmap="magma")
    ax.set_xlabel(r"$\cos \, \phi_{\rm Sgr}$")
    ax.set_ylabel(r"[Fe/H]")
    ax.text(0.1, 0.8, r"$L_y - L_z$ selected stars", transform=ax.transAxes)
    ax.set_ylim(*zrange)
    ax.axvline(plim, linestyle=":", color="k")
    ax = axes[2]
    ax.hist(rcat[arms & (phisgr < plim)]["FeH"], bins=20, range=zrange,
            density=True, alpha=0.5, orientation="horizontal")
    ax = axes[3]
    ax.hist(rcat[arms & (phisgr >= plim)]["FeH"], bins=20, range=zrange,
            density=True, alpha=0.5, orientation="horizontal")
    cbar = fig.colorbar(cb, cax=axes[0])
    axes[0].set_ylabel(r"$\chi_v$")
    axes[0].yaxis.set_ticks_position("left")
    axes[0].yaxis.set_label_position("left")

    fig.savefig("metal_angle.pdf")
    pl.close(fig)

    # --- FeH v Dist ---
    out = outlead | outtail
    dfig, daxes = pl.subplots(1, 2, sharey=True)
    dax = daxes[0]
    dax.plot(rcat[intail]["R_gal"], rcat[intail]["FeH"], color="maroon", 
             marker="o", linestyle="", alpha=0.5, label="Trailing")
    dax.plot(rcat[inlead]["R_gal"], rcat[inlead]["FeH"], color="slateblue", 
             marker="o", linestyle="", alpha=0.5, label="Leading")
    dax.plot(rcat[outtail]["R_gal"], rcat[outtail]["FeH"], color="maroon", 
             marker="o", linestyle="", markerfacecolor="white", label="Outliers (Trailing)")
    dax.plot(rcat[outlead]["R_gal"], rcat[outlead]["FeH"], color="slateblue", 
             marker="o", linestyle="", markerfacecolor="white", label="Outliers (Trailing)")
    dax.set_xlabel(r"$R_{\rm GC}$")
    dax.set_ylabel(r"[Fe/H]")

    dax = daxes[1]
    dax.plot(rcat[intail]["Sgr_l"], rcat[intail]["FeH"], color="maroon", 
             marker="o", linestyle="", alpha=0.5)
    dax.plot(rcat[inlead]["Sgr_l"], rcat[inlead]["FeH"], color="slateblue", 
             marker="o", linestyle="", alpha=0.5)
    dax.plot(rcat[outtail]["Sgr_l"], rcat[outtail]["FeH"], color="maroon", 
             marker="o", linestyle="", markerfacecolor="white")
    dax.plot(rcat[outlead]["Sgr_l"], rcat[outlead]["FeH"], color="slateblue", 
             marker="o", linestyle="", markerfacecolor="white")
    dax.set_xlabel(r"$\Lambda_{\rm Sgr}$")
    pl.close(dfig)


    # --- MDF ---
    figsize = 6, 6
    zfig, zaxes = pl.subplots(2, 1, sharey=True, sharex=True, figsize=figsize)
    zax = zaxes[0]
    tt = good & sel & trail
    n = tt.sum()
    med = np.median(rcat[tt]["FeH"])
    zax.hist(rcat[tt]["FeH"], bins=20, range=zrange, density=True,
             color="maroon", alpha=0.8, histtype="step", linewidth=3,
             label="Trailing, N={}".format(n))
    zax.set_ylim(0, zax.get_ylim()[-1]*1.1)
    yr = zax.get_ylim()
    zax.plot([med, med], yr[0] + np.diff(yr) * np.array([0.89, 0.96]), color="maroon", linewidth=2)
    zax.text(med+0.05, yr[0] + np.diff(yr) * 0.93, "{:3.2f}".format(med), color="maroon")

    ll = good & sel & lead
    n = ll.sum()
    med = np.median(rcat[ll]["FeH"])
    zax.hist(rcat[ll]["FeH"], bins=20, range=zrange, density=True, 
             color="slateblue", alpha=0.8, histtype="step", linewidth=3,
             label="Leading N={}".format(n))
    
    zax.plot([med, med], yr[0] + np.diff(yr) * np.array([0.89, 0.96]), color="slateblue", linewidth=2)
    zax.text(med-0.2, yr[0] + np.diff(yr) * 0.93, "{:3.2f}".format(med), color="slateblue")
    
    zax.legend(loc="upper left")
    #zax.set_xlabel("[Fe/H]")
    zax.text(0.05, 0.5, "All velocities", transform=zax.transAxes)


    zax = zaxes[1]
    zax.hist(rcat[intail]["FeH"], bins=20, range=zrange, density=True,
             color="maroon", alpha=0.8, histtype="step", linewidth=3,
             label="Trailing, N={}".format(intail.sum()))
    zax.hist(rcat[inlead]["FeH"], bins=20, range=zrange, density=True, 
             color="slateblue", alpha=0.8, histtype="step", linewidth=3,
             label="Leading, N={}".format(inlead.sum()))
    zax.hist(rcat[outlead | outtail]["FeH"], bins=20, density=True, color="orange",
             range=zrange, alpha=0.3,
             label="Velocity outliers N={}".format(outtail.sum() + outlead.sum()))
    #zax.hist(rcat[btout | blout]["FeH"], bins=20, density=True, color="orange",
    #         range=zrange, alpha=0.8, histtype="step", linewidth=3)
    zax.legend(loc="upper left")
    zax.set_xlabel("[Fe/H]")
    zax.text(0.05, 0.5, "Velocity outliers removed", transform=zax.transAxes)

    [ax.set_yticklabels([]) for ax in zaxes]
    [ax.set_ylabel("N([Fe/H]) (Normalized)") for ax in zaxes]
    zfig.savefig("figures/sgr_feh_{}_streams.{}".format(data_name, ext))

    pl.close(zfig)


    # --- Alpha-FeH ---
    figsize = 10, 5
    afig, aaxes = pl.subplots(1, 1, sharey=True, sharex=True, figsize=figsize)

    ax = aaxes
    ax.plot(rcat[outtail]["FeH"], rcat[outtail]["aFe"], 'o', color="maroon",
            alpha=0.5)
    ax.plot(rcat[outlead]["FeH"], rcat[outlead]["aFe"], 'o', color="slateblue",
            alpha=0.5)
    ax.plot(rcat[intail]["FeH"], rcat[intail]["aFe"], 'o', color="maroon",
            alpha=1.0, label="Leading")
    ax.plot(rcat[inlead]["FeH"], rcat[inlead]["aFe"], 'o', color="slateblue",
            alpha=1.0, label="Trailing")

    ax.legend()
    ax.set_xlabel(r"$[Fe/H]$")
    ax.set_ylabel(r"$[\alpha/Fe]$")

    afig.savefig("figures/afe_placeholder.{}".format(ext, dpi=300))
    pl.close(afig)
