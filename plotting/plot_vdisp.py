#!/usr/bin/python

"""Script to examine metallicity distributions in Sgr
"""

import sys

import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from utils import get_values, sgr_law10
from utils import read_lm, read_segue
from fit import Model
from outlier import read_from_h5

from matplotlib.pyplot import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'


# --- Literature traces ---
g17trail = [(240, 250, 260, 270),
            (15.7, 13.5, 12.5, 12.6), (1.5, 1.1, 1.1, 1.5),
            (14.0, 8.5, 7.1, 6.4), (2.6, 1.3, 1.4, 3.0),
            (-142.5, -124.2, -107.4, -79.0),
            (-143.5, -114.7, -98.6, -75.7)]
g17lead = [(105, 115, 125, 135),
           (31.4, 21.6, 15.0, 16.9), (6.0, 4.0, 4.0, 3.2),
           (19.7, 12.0, 6.0, 10.5), (5.0, 2.0, 1.5, 2.0),
           (-87.6, -107.2, -110.7, -123.8),
           (-77.5, -90.2, -107.1, -116.7)]

cols = ["lam", "vsig1", "vsig1_err", "vsig2", "vsig2_err", "vel1", "vel2"]
dt = np.dtype([(n, np.float) for n in cols])
gibt = np.zeros(len(g17trail[0]), dtype=dt)
for d, c in zip(g17trail, cols):
    gibt[c] = d

gibl = np.zeros(len(g17lead[0]), dtype=dt)
for d, c in zip(g17lead, cols):
    gibl[c] = d


belokurov14 = [(217.5, 227.5, 232.5, 237.5, 242.5, 247.5, 252.5, 257.5, 262.5,
                267.5, 272.5, 277.5, 285.0, 292.5),
               (-127.2, -141.1, -150.8, -141.9, -135.1, -129.5, -120.0, -108.8,
                -98.6, -87.2, -71.8, -58.8, -35.4, -7.8)]


if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = True
    show_lead = True

    htmodel, htresults = read_from_h5("vfit/h3_trail_vfit.h5")
    ltmodel, ltresults = read_from_h5("vfit/lm_trail_vfit.h5")
    hlmodel, hlresults = read_from_h5("vfit/h3_lead_vfit.h5")
    llmodel, llresults = read_from_h5("vfit/lm_lead_vfit.h5")

    # --- super figure ---
    # superfigure
    hcmap = "magma"
    fcolor = "slateblue"
    lmcolor = "orange"
    mlw = 3
    gcolors = "lightblue", "maroon"
    galpha = 0.5

    if show_lead:
        figsize = 8, 6
        ncol = 2
    else:
        figsize = 6, 6
        ncol = 1

    fig, axes = pl.subplots(2, ncol, sharex='col', sharey="row",
                            figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    vtax, stax = axes[:, 0]
    if show_lead:
        vlax, slax = axes[:, 1]

    # --- Plot H3 data ----
    vtax.set_ylim(-250, 50)
    vtax.set_ylabel(r"$V_{GSR}$")
    #vtax.yaxis.set_tick_params(which='both', labelbottom=True)

    # --- Plot fits ---
    tlam = np.arange(70, 140)
    llam = np.arange(220, 300)
    stax.set_xlabel(r"$\Lambda_{Sgr}$")
    stax.set_ylabel(r"$\sigma_v$")
    stax.set_ylim(-1, 30)

    # plot best fit to data on v-L and sigma-L plot
    imax = htresults["logl"].argmax()
    pmax = htresults["samples"][imax]
    mu, sigma = htmodel.model(tlam, pmax)
    out = htmodel.outlier_odds(pmax)
    vtax.plot(tlam, mu, color=fcolor, label="H3 fit")
    vtax.fill_between(tlam, mu - sigma, mu + sigma, alpha=0.5, color=fcolor)
    stax.plot(tlam, np.abs(sigma), color=fcolor, label="H3 fit")

    # Plot data
    out = out - np.median(out)
    tbh = vtax.scatter(htmodel.lamb, htmodel.vel, c=out, cmap=hcmap,
                       marker="o", edgecolors="k", linewidth=0.5,
                       alpha=0.5, s=16, vmin=out.min(), vmax=10)

    # plot the best fit for the LM data
    imax = ltresults["logl"].argmax()
    pmax = ltresults["samples"][imax]
    mu, sigma = ltmodel.model(tlam, pmax)
    stax.plot(tlam, np.abs(sigma), linestyle="--", linewidth=mlw,
              label="LM10 fit", color=lmcolor)
    vtax.plot(tlam, mu, linestyle="--", linewidth=mlw,
              label="LM10 fit", color=lmcolor)
    #lax.fill_between(ll, mu - sigma, mu + sigma, alpha=0.5)

    # leading data
    if show_lead:
        imax = hlresults["logl"].argmax()
        pmax = hlresults["samples"][imax]
        mu, sigma = hlmodel.model(llam, pmax)
        out = hlmodel.outlier_odds(pmax)
        vlax.plot(llam, mu, color=fcolor, label="H3 fit")
        vlax.fill_between(llam, mu - sigma, mu + sigma, alpha=0.5, color=fcolor)
        slax.plot(llam, np.abs(sigma), color=fcolor, label="H3 fit")

    # Plot data
        out = out - np.median(out)
        lbh = vlax.scatter(hlmodel.lamb, hlmodel.vel, c=out, cmap=hcmap,
                           edgecolors="k", linewidth=0.5, marker="o",
                           alpha=0.5, s=16, vmin=out.min(), vmax=10)

    # leading mock
        imax = llresults["logl"].argmax()
        pmax = llresults["samples"][imax]
        mu, sigma = llmodel.model(llam, pmax)
        slax.plot(llam, np.abs(sigma), linestyle="--", linewidth=mlw,
                  label="LM10 fit", color=lmcolor)
        vlax.plot(llam, mu, linestyle="--", linewidth=mlw,
                  label="LM10 fit", color=lmcolor)
        slax.set_xlabel(r"$\Lambda_{Sgr}$")

    # Gibbons17
        vlax.plot(360 - gibl["lam"], gibl["vel1"], label="G17 (Metal Poor)",
                  alpha=galpha, color=gcolors[0])
        vlax.plot(360 - gibl["lam"], gibl["vel2"], label="G17 (Metal Rich)",
                  alpha=galpha, color=gcolors[1])
        slax.errorbar(360 - gibl["lam"], gibl["vsig1"], yerr=gibl["vsig1_err"],
                      label="G17 (Metal Poor)", alpha=galpha, color=gcolors[0])
        slax.errorbar(360 - gibl["lam"], gibl["vsig2"], yerr=gibl["vsig2_err"],
                      label="G17 (Metal Rich)", alpha=galpha, color=gcolors[1])

    # Plot the gibson and belokurov trends
    vtax.plot(360 - gibt["lam"], gibt["vel1"], label="G17 (Metal Poor)",
              alpha=galpha, color=gcolors[0])
    vtax.plot(360 - gibt["lam"], gibt["vel2"], label="G17 (Metal Rich)",
              alpha=galpha, color=gcolors[1])

    stax.errorbar(360 - gibt["lam"], gibt["vsig1"], yerr=gibt["vsig1_err"],
                  label="G17 (Metal Poor)", alpha=galpha, color=gcolors[0])
    stax.errorbar(360 - gibt["lam"], gibt["vsig2"], yerr=gibt["vsig2_err"],
                  label="G17 (Metal Rich)", alpha=galpha, color=gcolors[1])

    # Monaco and Majewski
    stax.fill_between(np.array([30, 90]), 8.3 - 0.9, 8.3 + 0.9, alpha=0.5,
                      label="M07", color="royalblue")
    stax.plot(np.array([30, 90]), np.array([11.7, 11.7]), linestyle="--",
              label="M04", color="royalblue")
    stax.set_xlim(61, 149)

    stax.legend(fontsize=8)
    if savefigs:
        names = data_name, noisiness, selname, ext
        fig.savefig("figures/vfit_{}_{}_{}.{}".format(*names), dpi=300)
        pl.close(fig)
    else:
        pl.show(fig)
    sys.exit()

    # empirical velocity sidpersion of LM10 from fitted trand
    mu_pred, vpred = ltmodel.model(ltmodel.lamb, pmax)
    vpred = np.abs(vpred)
    vemp = ((ltmodel.vel - mu_pred)).std()

    # --- delta_v vs feh ---
    noisiness = "noisy"
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)
    rcat_vers = "1_4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))
    lm = read_lm(lmockfile)
    from astropy.io import fits
    rcat = fits.getdata(rcatfile)

    feh = rcat[htmodel.idx]["FeH"]
    pmax = htresults["samples"][np.argmax(htresults["logl"])]
    mu_pred, spred = htmodel.model(htmodel.lamb, pmax)
    dv = htmodel.vel - mu_pred
    g = slice(None)
    fig, ax = pl.subplots()
    ax.plot(dv[g], feh[g], "o", alpha=0.5)
