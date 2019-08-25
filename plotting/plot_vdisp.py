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


gibbons17 = [(240, 250, 260, 270), 
            (15.7, 13.5, 12.5, 12.6), (1.5, 1.1, 1.1, 1.5),
            (14.0, 8.5, 7.1, 6.4), (2.6, 1.3, 1.4, 3.0),
            (-142.5, -124.2, -107.4, -79.0),
            (-143.5, -114.7, -98.6, -75.7)]

cols = ["lam", "vsig1", "vsig1_err", "vsig2", "vsig2_err", "vel1", "vel2"]
dt = np.dtype([(n, np.float) for n in cols])
nbin = 4
gib = np.zeros(nbin, dtype=dt)
for d, c in zip(gibbons17, cols):
    gib[c] = d
    
belokurov14 = [(217.5, 227.5, 232.5, 237.5, 242.5, 247.5, 252.5, 257.5, 262.5, 267.5, 272.5, 277.5, 285.0, 292.5),
               (-127.2, -141.1, -150.8, -141.9, -135.1, -129.5, -120.0, -108.8, -98.6, -87.2, -71.8, -58.8, -35.4, -7.8)]


def read_from_h5(iname):
    rcols = ["logl", "samples", "logz", "logwt"]
    import h5py
    with h5py.File(iname, "r") as f:
        model = Model(f["alpha_range"][:], f["beta_range"][:], f["pout_range"][:])
        model.set_data(f["lamb"][:], f["vel"][:])
        results = {}
        for c in rcols:
            results[c] = f[c][:]

    return model, results


if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = True

    hmodel, hresults = read_from_h5("h3_trail_vfit.h5")
    lmodel, lresults = read_from_h5("lm_trail_vfit.h5")

    # --- super figure ---
    # superfigure
    hcmap = "magma"
    fcolor = "slateblue"
    lmcolor = "orange"
    gcolors = "green", "tomato"
    fig, axes = pl.subplots(2, 1, sharex=True)
    lax, vax = axes
    
    # --- Plot H3 data ----
    lbh = lax.plot(hmodel.lamb, hmodel.vel, color="maroon", marker="o", 
                   linestyle="", alpha=0.5, markersize=4)
    #cb = lfig.colorbar(lbh, ax=lax)
    #cb.set_label(z)
    lax.set_ylim(-300, 100)
    lax.set_ylabel(r"$V_{GSR}$")
    lax.yaxis.set_tick_params(which='both', labelbottom=True)


    # --- Plot fits ---
    ll = np.arange(70, 140)
    vax.set_xlabel(r"$\Lambda_{Sgr}$")
    vax.set_ylabel(r"$\sigma_v$")

    # plot best to data on v-L and sigma-L plot
    imax = hresults["logl"].argmax()
    pmax = hresults["samples"][imax]
    mu, sigma = hmodel.model(ll, pmax)
    lax.plot(ll, mu, color=fcolor, label="H3 fit")
    lax.fill_between(ll, mu - sigma, mu + sigma, alpha=0.5, color=fcolor)
    vax.plot(ll, np.abs(sigma), color=fcolor, label="H3 fit")

    # plot the best fit for the LM data
    imax = lresults["logl"].argmax()
    pmax = lresults["samples"][imax]
    mu, sigma = lmodel.model(ll, pmax)
    vax.plot(ll, np.abs(sigma), label="LM10 fit", color=lmcolor)
    lax.plot(ll, mu, label="LM10 fit", color=lmcolor)
    #lax.fill_between(ll, mu - sigma, mu + sigma, alpha=0.5)

    # Plot the gibson and belokurov trends
    lax.plot(360 - gib["lam"], gib["vel1"], label="Gib1", color=gcolors[0])
    lax.plot(360 - gib["lam"], gib["vel2"], label="Gib2", color=gcolors[1])

    vax.errorbar(360 - gib["lam"], gib["vsig1"], yerr=gib["vsig1_err"],
                 label="Gib1", color=gcolors[0])
    vax.errorbar(360 - gib["lam"], gib["vsig2"], yerr=gib["vsig2_err"],
                 label="Gib2", color=gcolors[1])

    vax.legend()
    if savefigs:
        fig.savefig("figures/vfit_placeholder.png", dpi=300)
        pl.close(fig)
    else:
        pl.show(fig)
    sys.exit()

    mu_pred, vpred = lmodel.model(lmodel.lamb, pmax)
    vpred = np.abs(vpred)
    vemp = ((lmodel.vel - mu_pred)).std()
