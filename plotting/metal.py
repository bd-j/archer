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

    trail = (rcat["Sgr_l"] < 150) & (rcat["V_gsr"] < 0)
    lead = (rcat["Sgr_l"] > 200) & (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140)

    # --- Vgsr lambda ----
    sel = good & ssel
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
    arms = good & ssel & (lead | trail)
    cb = ax.scatter(rcat[arms]["Sgr_l"], rcat[arms]["FeH"], c=rcat[arms]["V_gsr"], 
               alpha=0.8, vmin=-200, vmax=25)
    fig.colorbar(cb)
    pl.show()

    nvsig = 1
    lam = rcat["Sgr_l"]

    lmin = 40
    lmax = 320
    nbin = 14
    #bins = np.linspace(lmin, lmax, nbin + 1)
    bins = [50, 70, 80, 90, 100, 110, 120,
            220, 240, 260, 270, 290]
    bins = np.array(bins)
    nbin = len(bins) - 1
    binno = np.digitize(lam, bins, right=False)

    vbar, vsig = np.zeros(nbin), np.zeros(nbin)
    zbar, zsig = np.zeros(nbin), np.zeros(nbin)
    nz, nv = np.zeros(nbin), np.zeros(nbin)
    vsel = np.zeros_like(lam, dtype=bool)

    # --- Dumb fit ---
    for i in range(nbin):
        s = arms & (binno == i + 1)
        nv[i] = s.sum()
        if s.sum() < 5:
            continue
        rs = rcat[s]
        vbar[i] = rs["V_gsr"].mean()
        vsig[i] = rs["V_gsr"].std()

        gv = np.abs(rs["V_gsr"] - vbar[i]) / vsig[i] < nvsig
        nz[i] = gv.sum()
        vsel[s] = vsel[s] | gv
        if gv.sum() < 5:
            continue
        zbar[i] = rs[gv]["FeH"].mean()
        zsig[i] = rs[gv]["FeH"].std()

    #tfig, tax = pl.subplots()

    lax.errorbar((bins[1:] + bins[:-1]) / 2., vbar, vsig, marker="o")

    lead = (binno > 1) & (binno < 6)
    tail = (binno > 7) & (binno < 12)

    zfig, zax = pl.subplots()
    zax.hist(rcat[good & ssel & trail & vsel]["FeH"], bins=20, 
             range=(-2.8, 0.0), alpha=0.5, label="Trailing")
    zax.hist(rcat[good & ssel & lead & vsel]["FeH"], bins=20, 
             range=(-2.8, 0.0), alpha=0.5, label="Leading")
    zax.legend(loc=0)
    zax.set_xlabel("[Fe/H]")
    zfig.savefig("figures/sgr_feh_{}_streams.{}".format(data_name, ext))

    pl.show()