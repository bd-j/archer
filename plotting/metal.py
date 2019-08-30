#!/usr/bin/python

"""Script to examine metallicity distributions in Sgr
"""


import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from utils import get_values, sgr_law10
from utils import read_lm, read_segue
from plot_vdisp import read_from_h5

if __name__ == "__main__":

    ext = "png"
    segue_cat = False
    seguefile = "../data/catalogs/ksegue_gaia_v5.fits"
    rcat_vers = "1_4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))

    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    data_name = "H3v{}".format(rcat_vers)
    if segue_cat:
        rcat = read_segue(seguefile, rcat.dtype)
        data_name = "KSEG"

    # --- Quantity shortcuts ---
    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr_law10)
    feh = rcat["FeH"]

    # --- Basic selections ---
    # selections
    basic = ((rcat["FLAG"] == 0) & np.isfinite(rcat["Z_gal"]))
    giant = (rcat["logg"] < 3.5)
    extra = ((rcat["Vrot"] < 5) & (rcat["SNR"] > 3) &
             (rcat["BHB"] == 0) & (rcat["Teff"] < 7000) &
             (rcat["V_tan"] < 500))
    good = basic & giant & extra

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

    trail = (rcat["Sgr_l"] < 150)
    lead = (rcat["Sgr_l"] > 200)

    # Velocity selections
    htmodel, htresults = read_from_h5("h3_trail_vfit.h5")
    hlmodel, hlresults = read_from_h5("h3_lead_vfit.h5")

    tt = sel & good & trail
    htmodel.set_data(rcat["Sgr_l"][tt], rcat["V_gsr"][tt])
    pmax = htresults["samples"][np.argmax(htresults["logl"])]
    tout = htmodel.outlier_odds(pmax)
    tout = tout - np.median(tout)
    gtout = np.zeros_like(tt, dtype=bool)
    gtout[tt] = tout < 10

    ll = sel & good & lead
    hlmodel.set_data(rcat["Sgr_l"][ll], rcat["V_gsr"][ll])
    pmax = hlresults["samples"][np.argmax(hlresults["logl"])]
    lout = hlmodel.outlier_odds(pmax)
    lout = lout - np.median(lout)
    glout = np.zeros(len(rcat), dtype=bool)
    glout[ll] = lout < 10

    # select only velocity outliers
    blout = ll & (~glout)
    btout = tt & (~gtout)

    zrange = (-2.8, 0.0)

    # --- FeH vs Lambda ---
    fig, ax = pl.subplots()
    arms = good & sel & (lead | trail)
    cb = ax.scatter(rcat[arms]["Sgr_l"], rcat[arms]["FeH"],
                    c=rcat[arms]["V_gsr"], alpha=0.8, vmin=-200, vmax=25)
    fig.colorbar(cb)
    pl.show()
    pl.close(fig)

    # --- FeH vs cosphi ---
    plim = 0.97

    figsize = (15, 6)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, width_ratios=[0.2, 3, 1, 1],
                  left=0.05, right=0.95, wspace=0.25)
    axes = [fig.add_subplot(g) for g in gs]

    arms = good & (lead | trail)
    ax = axes[1]
    cb = ax.scatter(phisgr[arms], rcat[arms]["FeH"], c=lsgr[arms],
                    alpha=0.8, vmin=lslim, vmax=9000)
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
    axes[0].set_ylabel(r"$L_{\rm Sgr}$")
    axes[0].yaxis.set_ticks_position("left")
    axes[0].yaxis.set_label_position("left")

    pl.show()
    #pl.close(fig)

    import sys
    sys.exit()

    # --- MDF ---
    figsize = 6, 6
    zfig, zaxes = pl.subplots(2, 1, sharey=True, sharex=True, figsize=figsize)
    zax = zaxes[0]
    n = (good & sel & trail).sum()
    zax.hist(rcat[tt]["FeH"], bins=20, density=True, color="maroon",
             range=(-2.8, 0.0), alpha=0.3, label="Trailing, N={}".format(n))
    n = (good & sel & lead).sum()
    zax.hist(rcat[ll]["FeH"], bins=20, density=True, color="slateblue",
             range=(-2.8, 0.0), alpha=0.3, label="Leading N={}".format(n))
    zax.legend(loc="upper left")
    #zax.set_xlabel("[Fe/H]")
    zax.text(0.05, 0.5, "All velocities", transform=zax.transAxes)

    zax = zaxes[1]
    zax.hist(rcat[gtout]["FeH"], bins=20, density=True, color="maroon",
             range=zrange, alpha=0.3, label="Trailing, N={}".format(gtout.sum()))
    zax.hist(rcat[glout]["FeH"], bins=20, density=True, color="slateblue",
             range=zrange, alpha=0.3, label="Leading, N={}".format(glout.sum()))
    zax.hist(rcat[btout | blout]["FeH"], bins=20, density=True, color="orange",
             range=zrange, alpha=0.3,
             label="Velocity outliers N={}".format(btout.sum() + blout.sum()))
    zax.legend(loc="upper left")
    zax.set_xlabel("[Fe/H]")
    zax.text(0.05, 0.5, "Velocity outliers removed", transform=zax.transAxes)

    zfig.savefig("figures/sgr_feh_{}_streams.{}".format(data_name, ext))
    pl.close(zfig)

    # --- Alpha-FeH ---
    figsize = 10, 5
    afig, aaxes = pl.subplots(1, 1, sharey=True, sharex=True, figsize=figsize)

    ax = aaxes
    ax.plot(rcat[tt]["FeH"], rcat[tt]["aFe"], 'o', color="maroon",
            alpha=0.5)
    ax.plot(rcat[ll]["FeH"], rcat[ll]["aFe"], 'o', color="slateblue",
            alpha=0.5)
    ax.plot(rcat[gtout]["FeH"], rcat[gtout]["aFe"], 'o', color="maroon",
            alpha=1.0, label="Leading")
    ax.plot(rcat[glout]["FeH"], rcat[glout]["aFe"], 'o', color="slateblue",
            alpha=1.0, label="Trailing")

    ax.legend()
    ax.set_xlabel(r"$[Fe/H]$")
    ax.set_ylabel(r"$[\alpha/Fe]$")

    afig.savefig("figures/afe_placeholder.{}".format(ext, dpi=300))
    pl.show()
    pl.close(afig)
