#!/usr/bin/python

"""Script to examine metallicity distributions in Sgr
"""


import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from utils import get_values, sgr_law10
from utils import read_lm, read_segue

if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = False
    segue_cat = False

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


    # --- Vgsr lambda ----
    lfig, lax = pl.subplots()

    z, vmin, vmax = "FeH", -2.5, 0.0
    lbh = lax.scatter(rcat[sel]["Sgr_l"], rcat["V_gsr"][sel], 
                      c=rcat[z][sel], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbh, ax=lax)
    cb.set_label(z)

    lax.set_ylim(-300, 100)
    lax.set_xlabel(r"$\Lambda_{Sgr}$")
    lax.set_ylabel(r"$V_{GSR}$")
    lax.yaxis.set_tick_params(which='both', labelbottom=True)

    # --- FeH vs Lambda ---
    fig, ax = pl.subplots()
    arms = sel & (lead | trail)
    cb = ax.scatter(rcat[arms]["Sgr_l"], rcat[arms]["feh"],
                    c=rcat[arms]["V_gsr"],
                    alpha=0.8, vmin=-200, vmax=25)
    fig.colorbar(cb)
    pl.show()
    sys.exit()

    lam = rcat["Sgr_l"][sel & trail]
    vgsr = rcat["V_gsr"][sel & trail]

    # --- Smaht fit ---
    alpha_order, beta_order = 3, 2

    def model_lnlike(lam, vel, alpha, beta, pout=0.0, vmu_bad=-100, vsig_bad=200):
        vmu = np.dot(alpha[::-1], np.vander(lam, len(alpha)).T)
        vsig = np.dot(beta[::-1], np.vander(lam, len(beta)).T)

        norm = -np.log(np.sqrt(2 * np.pi * vsig**2))
        lnlike_good = norm - 0.5 * ((vel - vmu) / vsig)**2

        if pout > 0:
            bnorm = -np.log(np.sqrt(2 * np.pi * vsig_bad**2))
            lnlike_bad = bnorm - 0.5 * ((vel - vmu_bad) / vsig_bad)**2
            like = (1 - pout) * np.exp(lnlike_good) + pout * np.exp(lnlike_bad)
            return np.sum(np.log(like))
        else:
            return lnlike_good.sum()


    def lnprob(theta):
        lnprior = 0
        na, nb = alpha_order, beta_order
        alpha = theta[:na]
        beta = theta[na: (na + nb)]
        pout = theta[-1]

        lnlike = model_lnlike(lam, vgsr, alpha, beta, pout=pout)
        assert np.isfinite(lnlike), "{}".format(theta)
        return lnlike + lnprior


    alpha_min = np.array([0.,  -4., -0.5])
    alpha_max = np.array([200., 0.1, 0.5])
    beta_min = np.array( [-50., 0.])
    beta_max = np.array( [50.,  1.0])
    pout_min = np.array( [0.0])
    pout_max = np.array( [0.1])

    def prior_transform(u):
        na, nb = alpha_order, beta_order
        a = alpha_min[:na] + u[:na] * (alpha_max - alpha_min)[:na]
        b = beta_min + u[na:(na + nb)] * (beta_max - beta_min)
        pout = pout_min + u[-1] * (pout_max - pout_min)
        return np.hstack([a, b, pout])


    ndim = alpha_order + beta_order + 1
    import dynesty
    dsampler = dynesty.DynamicNestedSampler(lnprob, prior_transform, ndim)
    dsampler.run_nested()
    dresults = dsampler.results

    from dynesty import plotting as dyplot
    # Plot a summary of the run.
    rfig, raxes = dyplot.runplot(dresults)
    # Plot traces and 1-D marginalized posteriors.
    tfig, taxes = dyplot.traceplot(dresults)
    # Plot the 2-D marginalized posteriors.
    cfig, caxes = dyplot.cornerplot(dresults)
