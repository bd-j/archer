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
gib = np.zeros(4, dtype=dt)
for d, c in zip(gibbons, cols):
    gib[c] = d
    
belokurov14 = [(217.5, 227.5, 232.5, 237.5, 242.5, 247.5, 252.5, 257.5, 262.5, 267.5, 272.5, 277.5, 285.0, 292.5),
               (-127.2, -141.1, -150.8, -141.9, -135.1, -129.5, -120.0, -108.8, -98.6, -87.2, -71.8, -58.8, -35.4, -7.8)]

def dump_to_h5(results, model, oname):
    model_columns = ["alpha_range", "beta_range", "pout_range", "lamb", "vel"]
    import h5py
    with h5py.File(oname, "w") as out:
        for mc in model_columns:
            out.create_dataset(mc, data=model.__dict__[mc])
        for k, v in results.items():
            try:
                out.create_dataset(k, data=v)
            except:
                pass

if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = False
    segue_cat = False
    noisiness  = "noisy"

    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)
    seguefile = "../data/catalogs/ksegue_gaia_v5.fits"
    rcat_vers = "1_4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))

    # --- L & M 2010 model ---
    lm = read_lm(lmockfile)

    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    data_name = "H3v{}".format(rcat_vers)
    if segue_cat:
        rcat = read_segue(seguefile, rcat.dtype)
        data_name = "KSEG"

    # quantity shortcuts
    etot, lx, ly, lz, phisgr, lsgr = get_values(rcat, sgr=sgr_law10)
    lmq = get_values(lm, sgr=sgr_law10)
    etot_lm, lx_lm, ly_lm, lz_lm, phisgr_lm, lsgr_lm = lmq

    # --- Basic selections ---

    lmhsel = (lm["in_h3"] == 1)

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
    
    sel = good & sel

    trail = (rcat["Sgr_l"] < 150) & (rcat["V_gsr"] < 0)
    lead = (rcat["Sgr_l"] > 200) & (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140)


    # superfigure
    hcmap = "magma"
    fig, axes = pl.subplots(2, 1, sharex=True)
    lax, vax = axes

    # --- Vgsr lambda ----
    z, vmin, vmax = "FeH", -2.5, 0.0
    lbh = lax.scatter(rcat[sel]["Sgr_l"], rcat["V_gsr"][sel],
                      c=rcat[z][sel], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap=hcmap)
    cb = lfig.colorbar(lbh, ax=lax)
    cb.set_label(z)

    lax.set_ylim(-300, 100)
    #lax.set_xlabel(r"$\Lambda_{Sgr}$")
    lax.set_ylabel(r"$V_{GSR}$")
    lax.yaxis.set_tick_params(which='both', labelbottom=True)


    # --- Smaht fit ---
    # Set Priors and instantiate model---
    alpha_range = np.array([[ 100., -10., -0.2],
                            [1000.,  0.1,  0.2]])
    beta_range = np.array([[-50., 0., -0.1],
                           [ 50., 1.,  0.1]])
    pout_range = np.array([0, 0.1])

    # Instantiate model ---
    model = Model(alpha_range=alpha_range, beta_range=beta_range,
                  pout_range=pout_range)

    # Fit trailing data ---
    # select stars
    lam = rcat["Sgr_l"][sel & trail]
    vgsr = rcat["V_gsr"][sel & trail]
    model.set_data(lam, vgsr)
    from dynesty import DynamicNestedSampler as Sampler
    dsampler = Sampler(model.lnprob, model.prior_transform, model.ndim)
    dsampler.run_nested()
    h3results = dsampler.results
    dump_to_h5(h3results, model, "h3_trail_vfit.h5")

    from dynesty import plotting as dyplot
    cfig, caxes = dyplot.cornerplot(h3results)

    # Fit trailing mock ---
    # select stars
    msel = ((lm["Lmflag"] == -1) & (lm["Pcol"] < 3) &
            (lm["lambda"] < 125) & lmhsel)  # (lm["lambda"] > 25) )
    lmlam = lm["lambda"][msel]
    lmvgsr = lm["V_gsr"][msel]

    model.set_data(lmlam, lmvgsr)
    dsampler = Sampler(model.lnprob, model.prior_transform, model.ndim)
    dsampler.run_nested()
    lmresults = dsampler.results
    dump_to_h5(lmresults, model, "lm_trail_vfit.h5")


    # --- Plot fits ---
    ll = np.arange(70, 140)
    vax.set_xlabel(r"$\Lambda_{Sgr}$")
    vax.set_ylabel(r"$\sigma_v$")


    # plot best to data on v-L and sigma-L plot
    imax = h3results["logl"].argmax()
    pmax = h3results["samples"][imax]
    mu, sigma = model.model(ll, pmax)
    lax.plot(ll, mu, label="H3 fit")
    lax.fill_between(ll, mu - sigma, mu + sigma, alpha=0.5)
    vax.plot(ll, np.abs(sigma), label="H3 fit")

    # plot the best fit for the LM data
    imax = lmresults["logl"].argmax()
    pmax = lmresults["samples"][imax]
    mu, sigma = model.model(ll, pmax)
    vax.plot(ll, np.abs(sigma), label="LM10 fit")
    lax.plot(ll, mu, label="LM10 fit")
    #lax.fill_between(ll, mu - sigma, mu + sigma, alpha=0.5)

    # Plot the gibson and belokurov trends
    lax.plot(360 - gib["lam"], gib["vel1"], label="Gib1")
    lax.plot(360 - gib["lam"], gib["vel1"], label="Gib2")

    vax.errorbar(360 - gib["lam"], gib["vsig1"], yerr=gib["vsig1_err"], label="Gib1")
    vax.errorbar(360 - gib["lam"], gib["vsig2"], yerr=gib["vsig2_err"], label="Gib2")

    vax.legend()
    sys.exit()

    mfig, mlax = pl.subplots()
    mlax.plot(lmlam, lmvgsr, "o")
    imax = lmresults["logl"].argmax()
    pmax = lmresults["samples"][imax]
    mu, sigma = model.model(ll, pmax)
    mlax.plot(ll, mu)
    mlax.fill_between(ll, mu-sigma, mu+sigma, alpha=0.5, color="tomato")

    mu_pred, vpred = model.model(lmlam, pmax)
    vemp = ((lmvgsr - mu_pred)).std()
