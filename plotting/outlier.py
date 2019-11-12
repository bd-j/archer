#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from cornerplot import _hist2d as hist2d

from utils import get_values, sgr_law10
from utils import read_lm, read_segue
from fit import Model
#from plot_vdisp import read_from_h5


def read_from_h5(iname):
    rcols = ["logl", "samples", "logz", "logwt"]
    import h5py
    with h5py.File(iname, "r") as f:
        model = Model(f["alpha_range"][:], f["beta_range"][:],
                      f["pout_range"][:])
        try:
            idx = f["idx"][:]
        except:
            idx = [-1]
        model.set_data(f["lamb"][:], f["vel"][:], idx=idx)
        results = {}
        for c in rcols:
            results[c] = f[c][:]

    return model, results


def get_rcat_selections(rcat_vers="1_4", segue_cat=False, 
                        sgr=sgr_law10, lslim=1500, philim=0.75, 
                        lcut=[(-3500, -500), (4000, -6000)]):
    seguefile = "../data/catalogs/ksegue_gaia_v5.fits"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))

    # --- H3 ----
    rcat = fits.getdata(rcatfile)

    if segue_cat:
        rcat = read_segue(seguefile, rcat.dtype)
        data_name = "KSEG"

    # --- Quantity shortcuts ---
    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr)

    # --- Basic selections ---
    # selections
    basic = ((rcat["FLAG"] == 0) & np.isfinite(rcat["Z_gal"]))
    giant = (rcat["logg"] < 3.5)
    extra = ((rcat["Vrot"] < 5) & (rcat["SNR"] > 3) &
             (rcat["BHB"] == 0) & (rcat["Teff"] < 7000) &
             (rcat["V_tan"] < 500))

    # Sgr selections
    # Ly - Lz
    x, y = lcut
    m = np.diff(y) / np.diff(x)
    b = y[0] - m * x[0]
    m, b = m[0], b[0]
    lsel = lz < (m * ly + b)

    # phi - lsgr
    phisel = (phisgr > philim) & (lsgr > lslim)

    # etot -lsgr
    elim = -170000
    esel = (etot < 0) & (etot > elim)

    good = basic & giant & extra
    feh = rcat["FeH"]
    retro = (phisgr < -0.5) & (lsgr < -5000)

    return rcat, basic, giant, extra, lsel, phisel, esel


def delta_v(rcat):

    chiv = np.zeros(len(rcat))
    trail = (rcat["Sgr_l"] < 150)
    lead = (rcat["Sgr_l"] > 200)

    htmodel, htresults = read_from_h5("vfit/h3_trail_vfit.h5")
    hlmodel, hlresults = read_from_h5("vfit/h3_lead_vfit.h5")
    tmax = htresults["samples"][np.argmax(htresults["logl"])]
    lmax = hlresults["samples"][np.argmax(hlresults["logl"])]

    vmu, vsig = htmodel.model(rcat["Sgr_l"][trail], pos=tmax)
    chiv[trail] = (rcat["V_gsr"][trail] - vmu) / vsig

    vmu, vsig = hlmodel.model(rcat["Sgr_l"][lead], pos=lmax)
    chiv[lead] = (rcat["V_gsr"][lead] - vmu) / vsig

    return chiv


def vel_outliers(rcat, sel):

    trail = (rcat["Sgr_l"] < 150)
    lead = (rcat["Sgr_l"] > 200)

    htmodel, htresults = read_from_h5("vfit/h3_trail_vfit.h5")
    hlmodel, hlresults = read_from_h5("vfit/h3_lead_vfit.h5")
    tmax = htresults["samples"][np.argmax(htresults["logl"])]
    lmax = hlresults["samples"][np.argmax(hlresults["logl"])]

    # Velocity selections
    tt = sel & trail
    htmodel.set_data(rcat["Sgr_l"][tt], rcat["V_gsr"][tt])
    ptout = htmodel.outlier_odds(tmax)
    ptout = ptout - np.median(ptout)
    intail = np.zeros(len(tt), dtype=bool)
    intail[tt] = ptout < 10

    ll = sel & lead
    hlmodel.set_data(rcat["Sgr_l"][ll], rcat["V_gsr"][ll])
    plout = hlmodel.outlier_odds(lmax)
    plout = plout - np.median(plout)
    inlead = np.zeros(len(rcat), dtype=bool)
    inlead[ll] = plout < 10

    # select only velocity outliers
    outtail = tt & (~intail)
    outlead = ll & (~inlead)

    return intail, inlead, outtail, outlead


if __name__ == "__main__":
    segue_cat = False
    rcat_vers = "1_4"

    philim, lslim = 0.75, 1500
    x, y = [(-3500, -500), (4000, -6000)]
    blob = get_rcat_selections(segue_cat=segue_cat,
                               rcat_vers=rcat_vers,
                               sgr=sgr_law10,
                               philim=philim, lslim=lslim,
                               lcut=[x, y])
    rcat, basic, giant, extra, lsel, phisel, esel = blob

    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr_law10)
    ltot = np.sqrt(lx**2 + ly**2 + lz**2)
    feh = rcat["FeH"]

    good = basic & extra & giant
    sel = lsel & phisel & esel
    chiv = delta_v(rcat)
    intail, inlead, outtail, outlead = vel_outliers(rcat, good & sel)
    out = outlead | outtail

    zrange = (-2.8, 0.0)

    # FeH bump
    # show distributions for 
    #    full Sgr sample, 
    #    good stars that are very unaligned, 
    #    velocity outliers ? 
    #    moderatley aligned, lsgr > 1500 ?
    #    very aligned

    
    density = False
    fig, ax = pl.subplots()
    wh = good & sel & (phisgr < 1.)
    ax.hist(feh[wh], range=zrange, bins=20, density=density,
            alpha=0.9, histtype="step", linewidth=2,
            label=r"Sgr selection, $\cos\phi<${}, N={}".format(1., wh.sum()))
    llim = 0
    wh = good & (phisgr < philim) & (phisgr > 0.5) & (ltot > llim)
    ax.hist(feh[wh], range=zrange, bins=20, density=density,
            alpha=0.9, histtype="step", linewidth=2,
            label=r"Giants, $|L| > {}$, ${}<\cos\phi<${}, N={}".format(llim, 0.5, philim, wh.sum()))
    #wh = good & (phisgr > philim) & ~sel
    #ax.hist(feh[wh], range=zrange, bins=20, density=density,
    #        alpha=0.9, histtype="step", linewidth=2,
    #        label=r"Giants, $\cos\phi>${}, not selected, N={}".format(philim, wh.sum()))
    #wh = good & (phisgr > 0.95) & sel
    #ax.hist(feh[wh], range=zrange, bins=20, density=density,
    #        alpha=0.9, histtype="step", linewidth=2,
    #        label=r"Sgr, $\cos\phi>${} N={}".format(0.95, wh.sum()))
    wh = good & sel & out
    ax.hist(feh[wh], range=zrange, bins=20, density=density,
            alpha=0.9, histtype="step", linewidth=2,
            label=r"Sgr, $V_{GSR}$ outliers")
    ax.legend()
    pl.show()
    sys.exit()

    zfig, zax = pl.subplots()
    wh = good & ~sel
    zax.plot(phisgr[wh], feh[wh], marker="o", linestyle="", markersize=4, alpha=0.5, label="Not Sgr")
    wh = good & sel & ~out
    zax.plot(phisgr[wh], feh[wh], marker="o", linestyle="", markersize=4, alpha=0.6, label="Sgr inliers")
    wh = good & sel & out
    zax.plot(phisgr[wh], feh[wh], marker="o", linestyle="", markersize=4, alpha=0.8, label="Sgr outliers")
    zax.legend()

    #mdf = np.histogram2d(phisgr[wh], feh[wh])

    lowzout = (out) & (feh < -1.8)
    with open("outliers.txt", "w") as fout:
        fout.write("h3id  lambda  FeH\n")
        for row in rcat[lowzout]:
            line = "{}   {:4.2f}  {:3.2f}\n".format(row["H3_ID"], row["Sgr_l"], row["FeH"])
            fout.write(line)


