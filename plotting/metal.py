#!/usr/bin/python

"""Script to examine metallicity distributions in Sgr
"""


import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from utils import get_values, sgr_law10
from utils import read_lm, read_segue

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
    ssel, selname = phisel & lsel & esel, "allsel"

    trail = (rcat["Sgr_l"] < 150) #& (rcat["V_gsr"] < 0)
    lead = (rcat["Sgr_l"] > 200) #& (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140)

    tvsel = (rcat["V_gsr"] < 0)
    lvsel = (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140)

    # --- FeH vs Lambda ---
    fig, ax = pl.subplots()
    arms = good & ssel & (lead | trail)
    cb = ax.scatter(rcat[arms]["Sgr_l"], rcat[arms]["FeH"], c=rcat[arms]["V_gsr"], 
               alpha=0.8, vmin=-200, vmax=25)
    fig.colorbar(cb)
    pl.show()

    zfig, zaxes = pl.subplots(1, 2, sharey=True)
    zax = zaxes[0]
    zax.hist(rcat[good & ssel & trail]["FeH"], bins=20, #density=True,
             range=(-2.8, 0.0), alpha=0.5, label="Trailing")
    zax.hist(rcat[good & ssel & lead]["FeH"], bins=20, #density=True,
             range=(-2.8, 0.0), alpha=0.5, label="Leading")
    zax.legend(loc=0)
    zax.set_xlabel("[Fe/H]")
    zax.set_title("All velocities")
    
    zax = zaxes[1]
    zax.hist(rcat[good & ssel & trail & tvsel]["FeH"], bins=20, #density=True,
             range=(-2.8, 0.0), alpha=0.5, label="Trailing")
    zax.hist(rcat[good & ssel & lead & lvsel]["FeH"], bins=20, #density=True,
             range=(-2.8, 0.0), alpha=0.5, label="Leading")
    zax.legend(loc=0)
    zax.set_xlabel("[Fe/H]")
    zax.set_title("Crude velocity cut")

    
    #zfig.savefig("figures/sgr_feh_{}_streams.{}".format(data_name, ext))

    pl.show()