#!/usr/bin/python

"""Script to calculate the angular offset between H3 stars and the 
Sagittarius dwarf galaxy angular momentum vector. Do it for mocks as well.
"""

import sys
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
import gala

from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc

from utils import read_lm, read_segue
from utils import lm_quiver, h3_quiver, overplot_clump
from utils import get_Lsgr, compute_Lstar, angle_to_sgr, gsr_to_rv
from utils import gc_frame_law10, sgr_law10, sgr_fritz18


rcParams["font.family"] = "serif"


if __name__ == "__main__":

    # choose file format
    ext = "png"  # "pdf" | "png"
    savefigs = False
    segue_cat = False
    noisiness = "noisy"  # "noisy" | "noiseless"

    seguefile = "../data/catalogs/ksegue_gaia_v5.fits"
    rcat_vers = "1_4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers.replace("_", "."))
    lmockfile = "../data/mocks/LM10/LM10_15deg_{}_v5.fits".format(noisiness)

    # --- L & M 2010 model ---
    lm = read_lm(lmockfile)
    Lsgr_lm10 = get_Lsgr(sgr_law10, gc_frame=gc_frame_law10)
    vlos = gsr_to_rv(lm["v"], lm["ra"], lm["dec"], lm["dist"])

    lstar_lm10 = compute_Lstar(lm["ra"], lm["dec"], lm["dist"], 
                               lm["mua"], lm["mud"], vlos.value,
                               gc_frame=gc_frame_law10)
    lmTheta, lmProj = angle_to_sgr(Lsgr_lm10, lstar_lm10)

    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    data_name = "H3v{}".format(rcat_vers)
    if segue_cat:
        rcat = read_segue(seguefile, rcat.dtype)
        data_name = "KSEG"
    Lsgr_h3 = get_Lsgr(sgr_law10, gc_frame=gc_frame_law10)
    lstar_h3 = compute_Lstar(rcat["RA"], rcat["DEC"], rcat["dist_adpt"], 
                             rcat["GaiaDR2_pmra"], rcat["GaiaDR2_pmdec"], 
                             rcat["Vrad"],
                             gc_frame=gc_frame_law10)
    h3Theta, h3Proj = angle_to_sgr(Lsgr_h3, lstar_h3)


    # --- Basic selections ---
    # quantity shortcuts
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
    feh = np.clip(rcat["FeH"], -3.0, 0.1)
    sgr_lowb = np.abs(rcat["Sgr_b"]) < 40
    etot = rcat["E_tot_pot2"]
    theta_thresh = 0.75

    # selections
    feh_sel = (rcat["BHB"] == 0) & (rcat["Teff"] < 7000) & (rcat["V_tan"] < 500)
    good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
            (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
    extra = (rcat["Vrot"] < 5) & (rcat["SNR"] > 3)
    good = good & extra
    aligned = (h3Theta > theta_thresh)
    far = rcat["R_gal"] > 7

    # Subsets of LM10
    np.random.seed(101)
    linds = np.random.choice(len(lm), size=int(0.1 * len(lm)))
    lmsel = np.zeros(len(lm), dtype=bool)
    lmsel[linds] = True
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(0, 1, len(lm)) < 0.5) #& (lmTheta > theta_thresh)

    # special selection
    clump = (
             #(h3Proj > 1000) & (h3Proj < 4000) &
             #(etot > -175000) & (etot < -140000) &
             (h3Proj > 5000) & (h3Proj < 10000) & 
             (etot > -88500) & (etot < -75500) &
             aligned & good
             )


    # --- Phi_sgr histograms ---
    dfig, dax = pl.subplots()
    peri = np.unique(lm["Pcol"])
    for p in peri:
        psel = lmhsel & (lm["Pcol"] == p)
        dax.hist(lmTheta[psel], bins=100, range=(-1, 1), 
                 alpha=0.5, label="LM10 peri #{}".format(peri.max()-p))
    hfig, hax = pl.subplots()
    hax.hist(h3Theta[good & far], bins=100, range=(-1, 1), alpha=0.3, color="maroon", label=data_name)
    [ax.set_xlabel(r"$\cos \phi_{Sgr}$") for ax in [hax, dax]]
    [ax.legend(loc="upper left") for ax in [hax, dax]]

    if savefigs:
        dfig.savefig("figures/angle_distribution_lm10{}.{}".format(noisiness, ext))
        hfig.savefig("figures/angle_distribution_{}.{}".format(data_name, ext))
        [pl.close(f) for f in [hfig, dfig]]


    # --- Quiver plots ---
    lcatz = [lm["Lmflag"], lm["Lmflag"]]
    rcatz = [feh, feh]
    ascale = 25
    nrow, ncol = 2, 3
    projections = ["xz", "xy"]

    from matplotlib import cm, colors
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=-3., vmax=3., clip=False)

    sel = good & aligned & far
    fig, axes = pl.subplots(nrow, ncol, sharex=True, sharey="row", figsize=(24, 12.5))
    cbl = [lm_quiver(lm[lmsel], z[lmsel], vtot=vtot_lm[lmsel], show=s, ax=ax, scale=ascale, cmap=cmap, norm=norm)
           for s, ax, z in zip(projections, axes[:, 0], lcatz)]
    cblh = [lm_quiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel], show=s, ax=ax, scale=ascale, cmap=cmap, norm=norm)
            for s, ax, z in zip(projections, axes[:, 1], lcatz)]
    cbh = [h3_quiver(rcat[sel], z[sel], vtot=vtot[sel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, axes[:, 2], rcatz)]

    axes[0, 0].set_xlim(-70, 40)
    axes[0, 0].set_ylim(-80, 80)
    axes[1, 0].set_ylim(-30, 40)
    [ax.xaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[0, :]]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[:, 1:].flat]

    axes[0, 0].set_title("LM10")
    axes[0, 1].set_title("LM10xH3")
    axes[0, 2].set_title(data_name)
    for i in range(nrow):
        xl, yl = projections[i]
        for j in range(ncol):
            axes[i, j].set_xlabel(xl.upper())
            axes[i, j].set_ylabel(yl.upper())

    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())

    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    #axes = overplot_clump(axes, rcat[clump], clumpcolor="maroon")

    if savefigs:
        fig.savefig("figures/quiver_lm10{}_{}_phisel.{}".format(noisiness, data_name, ext), dpi=300)
        pl.close(fig)

    # --- Bifurcation ---
    stream = good & (rcat["Sgr_l"] > 200) & (rcat["Sgr_l"] < 300)
    A = stream & (rcat["Sgr_b"] > 0)
    B = stream & (rcat["Sgr_b"] < 0)

    # --- E-Lsgr ---
    efig, eaxes = pl.subplots(1, 2, sharey=True)
    eax = eaxes[1]
    eax.plot(h3Proj[good & ~sel], rcat[good & ~sel]["E_tot_pot2"], 'o', alpha=0.4, label="unaligned")
    eax.plot(h3Proj[sel], rcat[sel]["E_tot_pot2"], 'o', alpha=0.4, label="aligned")
    eax.legend()
    eax = eaxes[0]
    ec = eax.scatter(lmProj[lmhsel], lm[lmhsel]["E_tot_pot2"], c=lm[lmhsel]["Lmflag"], 
                     vmin=-2, vmax=3, marker='o', alpha=0.4)
    #cb = efig.colorbar(ec, ax=eax, location="top")
    #cb.set_label("Lmflag")
    [a.set_ylim(-2e5, -5e4) for a in eaxes]
    [a.set_xlim(-1e4, 2e4) for a in eaxes]
    [a.set_xlabel(r"$L_{sgr}$") for a in eaxes]
    [a.set_ylabel(r"$E_{tot}$") for a in eaxes]

    if savefigs:
        pl.close(efig)


    # --- Vgsr lambda ----
    sel = good & aligned

    lfig, laxes = pl.subplots(1, 3, sharex=True, sharey=True,
                              constrained_layout=True, figsize=(16, 4))

    lax = laxes[0]
    z, vmin, vmax = "Lmflag", -2, 3
    lbm = lax.scatter(lm[lmsel]["lambda"], lm[lmsel]["V_gsr"], 
                      c=lm[lmsel][z], marker=".", s=2,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbm, location="top", ax=lax)
    cb.set_label(z)

    lax = laxes[1]
    lbm = lax.scatter(lm[lmhsel]["lambda"], lm[lmhsel]["V_gsr"], 
                      c=lm[lmhsel][z], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbm,  location="top", ax=lax)
    cb.set_label(z)

    lax = laxes[2]
    z, vmin, vmax = "FeH", -2.5, 0.0
    lbh = lax.scatter(rcat[sel]["Sgr_l"], rcat["V_gsr"][sel], 
                      c=rcat[z][sel], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbh, location="top", ax=lax)
    cb.set_label(z)


    lax.set_ylim(-300, 300)
    laxes[0].set_title("LM10")
    laxes[1].set_title("LM10xH3")
    laxes[2].set_title("H3")

    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in laxes]
    [ax.set_ylabel(r"$V_{GSR}$") for ax in laxes]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
     for ax in laxes[1:]]
    [ax.invert_xaxis() for ax in laxes]

    if savefigs:
        lfig.savefig("figures/vgsr_Lambda_{}_{}_phisel.{}".format(data_name, noisiness, ext), dpi=300)
        pl.close(lfig)


    # --- distance - lambda ----
    sel = good & aligned
    vlim = -400, 200

    lfig, laxes = pl.subplots(1, 3, sharex=True, sharey=True,
                              constrained_layout=True, figsize=(16, 4))

    lax = laxes[0]
    z, vmin, vmax = "Lmflag", -2, 3
    lbm = lax.scatter(lm[lmsel]["lambda"], lm[lmsel]["R_gal"], 
                      c=lm[lmsel][z], marker=".", s=2,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbm, location="top", ax=lax)
    cb.set_label(z)

    lax = laxes[1]
    z, vmin, vmax = "vgsr", -400, 200
    lbm = lax.scatter(lm[lmhsel]["lambda"], lm[lmhsel]["R_gal"], 
                      c=lm[lmhsel][z], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbm, location="top", ax=lax)
    cb.set_label(z)


    lax = laxes[2]
    z, vmin, vmax = "V_gsr", -300, 200
    lbh = lax.scatter(rcat[sel]["Sgr_l"], rcat["R_gal"][sel], 
                      c=rcat[z][sel], marker="o", alpha=0.7, s=4,
                      vmin=vmin, vmax=vmax, cmap="viridis")
    cb = lfig.colorbar(lbh,  location="top", ax=lax)
    cb.set_label(z)


    lax.set_ylim(0, 100)
    laxes[0].set_title("LM10")
    laxes[1].set_title("LM10xH3")
    laxes[2].set_title("H3")

    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in laxes]
    laxes[0].set_ylabel(r"$R_{gal}$")
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
     for ax in laxes[1:]]
    [ax.invert_xaxis() for ax in laxes]

    if savefigs:
        lfig.savefig("figures/Rgal_Lambda_{}_{}_phisel.{}".format(data_name, noisiness, ext), dpi=300)
        pl.close(lfig)

    # --- FeH ---
    trail = (good & aligned & 
             (rcat["Sgr_l"] < 150) & (rcat["V_gsr"] < 0) &
             (etot > -150000))

    lead = (good & aligned & 
             (rcat["Sgr_l"] > 200) & (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140) &
             (etot > -150000))

    zfig, zax = pl.subplots()
    zax.hist(feh[trail], bins=30, alpha=0.5, label="Trailing")
    zax.hist(feh[lead], bins=30, alpha=0.5, label="Leading")
    zax.set_xlim(-3, 0.0)
    zax.legend()
    zax.set_xlabel(r"[Fe/H]")

    if savefigs:
        zfig.savefig("figures/sgr_feh_{}_streams.{}".format(data_name, ext), dpi=300)
        pl.close(zfig)


    # --- polar sky position ---
    sel = good & aligned
    pfig = pl.figure()
    pax = pl.subplot(projection="polar")
    cb = pax.plot(np.deg2rad(360 - rcat[sel]["Sgr_l"]), rcat[sel]["dist_adpt"],
                  marker='o', linestyle="", markersize=2, alpha=0.7)
                  # c=rcat[sel]["Sgr_b"], vmin=-20, vmax=20)
    pax.set_rmax(100)
    lfig = pl.figure()
    lax = pl.subplot(projection="polar")
    lax.scatter(np.deg2rad(lm[lmsel]["Sgr_Lam"]), lm[lmsel]["dist"], c=lm[lmsel]["Lmflag"],
                marker='o', vmin=-3, vmax=3, alpha=0.4, s=2)
    cb = lax.scatter(np.deg2rad(lm[lmhsel]["Sgr_Lam"]), lm[lmhsel]["dist"],
                     c=lm[lmhsel]["Lmflag"], marker='o', vmin=-3, vmax=3, alpha=0.8, s=5)
    lax.set_rmax(100)


    pl.show()
    sys.exit()

    # --- sky position ---

    sel = good & aligned
    sfig, saxes = pl.subplots(2, 2, sharey="row", sharex="col")
    saxes[0, 0].scatter(lm[lmsel]["lambda"], lm[lmsel]["beta"], c=lm["Pcol"][lmsel], 
                        vmin=-1, vmax=9.0, alpha=0.5)
    saxes[0, 1].scatter(rcat[sel]["Sgr_l"], rcat[sel]["sgr_b"], c=h3Theta[sel], 
                        vmin=theta_thresh, vmax=1.0, alpha=0.5)

    saxes[1, 0].scatter(lm[lmsel]["l"], lm[lmsel]["b"], c=lm["Pcol"][lmsel], 
                        vmin=-1, vmax=9, alpha=0.5)
    saxes[1, 1].scatter(rcat[sel]["L"], rcat[sel]["B"], c=h3Theta[sel], 
                        vmin=theta_thresh, vmax=1.0, alpha=0.5)

    [ax.set_ylim(-50, 50) for ax in saxes[0, :]]
    [ax.set_ylim(-90, 90) for ax in saxes[1, :]]
    [ax.set_ylabel("B") for ax in saxes[0, :]]
    [ax.set_ylabel("b") for ax in saxes[1, :]]
    [ax.set_xlabel(r"$\Lambda$") for ax in saxes[0, :]]
    [ax.set_xlabel(r"$\ell$") for ax in saxes[1, :]]
