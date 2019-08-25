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
    #from dynesty import plotting as dyplot
    #cfig, caxes = dyplot.cornerplot(h3results)

    # Fit leading data ---
    # select stars
    lam = rcat["Sgr_l"][sel & lead]
    vgsr = rcat["V_gsr"][sel & lead]
    model.set_data(lam, vgsr)
    from dynesty import DynamicNestedSampler as Sampler
    dsampler = Sampler(model.lnprob, model.prior_transform, model.ndim)
    dsampler.run_nested()
    h3results = dsampler.results
    dump_to_h5(h3results, model, "h3_lead_vfit.h5")


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

    sys.exit()

    # Fit leading mock ---
    # select stars
    msel = ((lm["Lmflag"] == 1) & (lm["Pcol"] < 6) &
            (lm["lambda"] > 200) & lmhsel)  # (lm["lambda"] > 25) )
    lmlam = lm["lambda"][msel]
    lmvgsr = lm["V_gsr"][msel]

    model.set_data(lmlam, lmvgsr)
    dsampler = Sampler(model.lnprob, model.prior_transform, model.ndim)
    dsampler.run_nested()
    lmresults = dsampler.results
    dump_to_h5(lmresults, model, "lm_lead_vfit.h5")


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
