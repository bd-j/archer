import sys
import numpy as np
import matplotlib.pyplot as pl
pl.ion()
from astropy.io import fits

from utils import h3_quiver, lm_quiver, read_lm, get_sgr


if __name__ == "__main__":
    lmocks = ["../data/mocks/LM10/SgrTriax_DYN.dat",
              "../data/mocks/LM10/LM10_h3_noiseless_v1.fits",
              "../data/mocks/LM10/LM10_h3_noisy_v1.fits"]
    rcat_vers = "1.4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers)
    lmockfile, noisiness = lmocks[2], "noisy"

    # --- L & M 2010 model ---

    lm = read_lm(lmockfile)
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)
    #vtot_lm = np.sqrt(lm["Vx_gal"]**2 + lm["Vy_gal"]**2 + lm["Vz_gal"]**2)
    np.random.seed(101)
    lmsel = np.random.choice(len(lm), size=int(0.1 * len(lm)))
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(0, 1, len(lm)) < 0.5)

    lmy = np.clip(lm["Y_gal"], -40, 40)
    lmz = np.clip(lm["Z_gal"], -80, 80)
    lmf = lm["Lmflag"]

    # --- RCAT and associated quantities ---
    rcat = fits.getdata(rcatfile)
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
    vy, vz = rcat["Vy_gal"] / vtot, rcat["Vz_gal"] / vtot
    ygal = np.clip(rcat["Y_gal"], -40, 40)
    zgal = np.clip(rcat["Z_gal"], -80, 80)
    feh = np.clip(rcat["feh"], -2.0, 0.1)
    IDs = np.unique(rcat["selID"]).tolist()
    selID = np.array([IDs.index(s) for s in rcat["selID"]])

    # --- Basic selections ---
    good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
            (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
    sgr = np.abs(rcat["Sgr_B"]) < 40
    etot = rcat["E_tot_pot1"]
    sel = good #& (etot > -130000) & (etot < -50000)
    sel = np.arange(len(sel))[sel]
    rind = np.random.permutation(len(sel))
    sel = sel[rind]

    # --- Quiver ---
    nrow, ncol = 2, 3
    projections = ["xz", "xy"]
    ascale = 25

    lcatz = [lmy, lmz]
    #lcatz = [lmf, lmf]
    #lcatz = [lm["v"]/vtot_lm, lm["w"]/vtot_lm]
    rcatz = [ygal, zgal]
    #rcatz = [feh, feh]
    #rcatz = [np.clip(v, -1, 1) for v in [vy, vz]]

    fig, axes = pl.subplots(nrow, ncol, sharex=True, sharey="row", figsize=(24, 12.5))
    cbl = [lm_quiver(lm[lmsel], z[lmsel], vtot=vtot_lm[lmsel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, axes[:, 0], lcatz)]
    cblh = [lm_quiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel], show=s, ax=ax, scale=ascale)
            for s, ax, z in zip(projections, axes[:, 1], lcatz)]
    cbh = [h3_quiver(rcat[sel], z[sel], vtot=vtot[sel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, axes[:, 2], rcatz)]

    axes[0, 0].set_xlim(-90, 40)
    axes[0, 0].set_ylim(-80, 100)
    axes[1, 0].set_ylim(-40, 60)

    [ax.xaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[0, :]]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[:, 1:].flat]

    axes[0, 0].set_title("LM10")
    axes[0, 1].set_title("LM10xH3")
    axes[0, 2].set_title("H3")
    for i in range(nrow):
        xl, yl = projections[i]
        for j in range(ncol):
            axes[i, j].set_xlabel(xl.upper())
            axes[i, j].set_ylabel(yl.upper())

    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())

    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    fig.savefig("figures/sgr_lm10_h3v{}_{}.png".format(rcat_vers, noisiness), dpi=150)
    
    sys.exit()

    # ----- Vr Vtheta ----
    np.random.seed(101)
    rind = np.random.permutation(len(lm))
    vtype = "Vtheta_gal"
    vfig, vaxes = pl.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(22, 6))

    lcatz = np.clip(lm["R_gal"], 5, 80)
    rcatz = np.clip(rcat["R_gal"], 5, 80)
    vax = vaxes[0]
    vbm = vax.scatter(lm[rind]["Vr_gal"], lm[rind][vtype], 
                    c=lcatz[rind], marker=".", s=1,
                    cmap="viridis")
    vax = vaxes[1]
    vbm = vax.scatter(lm[lmhsel]["Vr_gal"], lm[lmhsel][vtype], 
                    c=lcatz[lmhsel], marker="o", alpha=0.7, s=4,
                    cmap="viridis")

    vax = vaxes[2]
    vbh = vax.scatter(rcat[good & sgr]["Vr_gal"], rcat[vtype][good & sgr], 
                    c=rcatz[good & sgr], marker="o", alpha=0.7, s=4,
                    cmap="viridis")
    vax.set_xlim(-300, 300)
    vax.set_ylim(-500, 450)

    vaxes[0].set_title("LM10")
    vaxes[1].set_title("LM10xH3")
    vaxes[2].set_title("H3")

    [ax.set_xlabel(r"$V_r$") for ax in vaxes]
    [ax.set_ylabel(r"$V_\theta$") for ax in vaxes]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
    for ax in vaxes[1:]]

    vfig.colorbar(vbh, ax=vaxes)

    vfig.savefig("figures/sgr_vrvt_lm10_h3.png", dpi=300)

    sys.exit()

    # --- Vgsr lambda ----

    lfig, laxes = pl.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(16, 4))

    lax = laxes[0]
    lbm = lax.scatter(lm[rind]["lambda"], lm[rind]["V_gsr"], 
                    c=lm[rind]["Lmflag"], marker=".", s=1,
                    vmin=-2, vmax=3, cmap="viridis")
    lax = laxes[1]
    lbm = lax.scatter(lm[lmhsel]["lambda"], lm[lmhsel]["V_gsr"], 
                    c=lm[lmhsel]["Lmflag"], marker="o", alpha=0.7, s=4,
                    vmin=-2, vmax=3, cmap="viridis")

    lax = laxes[2]
    lbh = lax.scatter(rcat[good & sgr]["sgr_l"], rcat["V_gsr"][good & sgr], 
                    c=rcat["dist_adpt"][good & sgr], marker="o", alpha=0.7, s=4,
                    vmin=0, vmax=100, cmap="viridis")

    lax.set_ylim(-350, 350)
    laxes[0].set_title("LM10")
    laxes[1].set_title("LM10xH3")
    laxes[2].set_title("H3")

    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in laxes]
    [ax.set_ylabel(r"$V_{GSR}$") for ax in laxes]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
    for ax in laxes[1:]]

    lfig.colorbar(lbh, ax=laxes)
    lfig.savefig("figures/sgr_lvgsr_lm10_h3.png")


    # --- beta lambda ----

    csgr = get_sgr(rcat[sel])

    lfig, laxes = pl.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(16, 4))

    lax = laxes[0]
    lbm = lax.scatter(lm[rind]["lambda"], lm[rind]["beta"], 
                    c=lm[rind]["V_gsr"], marker=".", s=1,
                    vmin=-350, vmax=350, cmap="viridis")
    lax = laxes[1]
    lbm = lax.scatter(lm[lmhsel]["lambda"], lm[lmhsel]["beta"], 
                    c=lm[lmhsel]["V_gsr"], marker="o", alpha=0.7, s=4,
                    vmin=-350, vmax=350, cmap="viridis")

    lax = laxes[2]
    lbh = lax.scatter(csgr.Lambda, csgr.Beta, 
                    c=rcat[sel]["V_gsr"], marker="o", alpha=0.7, s=4,
                    vmin=-350, vmax=350, cmap="viridis")

    #lax.set_ylim(-, 350)
    laxes[0].set_title("LM10")
    laxes[1].set_title("LM10xH3")
    laxes[2].set_title("H3")

    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in laxes]
    [ax.set_ylabel(r"$B_{Sgr}$") for ax in laxes]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
    for ax in laxes[1:]]

    lfig.colorbar(lbh, ax=laxes)
    lfig.savefig("figures/sgr_lb_lm10_h3.png", dpi=300)