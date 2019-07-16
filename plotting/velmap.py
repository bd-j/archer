import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

vmap = {"x": "u",
        "y": "v",
        "z": "w"}

def h3_quiver(cat, zz, vtot=1.0, show="xy", ax=None):
    x, y = [cat["{}_gal".format(s.upper())] for s in show]
    vx, vy = [cat["V{}_gal".format(s)] for s in show]
    cb = ax.quiver(x, y, vx / vtot, vy / vtot, zz,
                   angles="xy", pivot="mid", cmap="viridis",
                   scale_units="height", scale=20)

    return cb


def lm_quiver(cat, zz, vtot=1.0, show="xy", ax=None):
    x, y = [cat["{}gc".format(s)] for s in show]
    vx, vy = [cat[vmap[s]] for s in show]
    cb = ax.quiver(x, y, vx / vtot, vy / vtot, zz,
                   angles="xy", pivot="mid", cmap="viridis",
                   scale_units="height", scale=20)

    return cb


# --- L & M 2010 model ---
lmfile = "../mocks/Sgr/SgrTriax_DYN.dat"

with open(lmfile, "r") as f:
    cols = f.readline().split()
    dt = np.dtype([(c, np.float) for c in cols])
lm = np.genfromtxt(lmfile, skip_header=1, dtype=dt)
# Make right handed
lm["xgc"] = -lm["xgc"]
lm["u"] = -lm["u"]
vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2, lm["w"]**2)
lmsel = np.random.choice(len(lm), size=int(0.1 * len(lm)))
lmy = np.clip(lm["ygc"], -60, 100)
lmz = np.clip(lm["ygc"], -80, 100)
lmf = lm["Lmflag"]

# --- RCAT and associated quantities ---
rcat = fits.getdata("rcat_V1.3_MSG.fits")

vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
zgal = np.clip(rcat["Z_gal"], -100, 100)
ygal = np.clip(rcat["Y_gal"], -60, 100)
feh = np.clip(rcat["feh"], -3.0, 0.1)
IDs = np.unique(rcat["selID"]).tolist()
selID = np.array([IDs.index(s) for s in rcat["selID"]])

# --- Basic selections ---
good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
        (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
sgr = np.abs(rcat["Sgr_B"]) < 13
etot = rcat["E_tot_pot1"]
sel = good #& (etot > -130000) & (etot < -50000)

# --- Histograms ---
#fig, axes = pl.subplots(2, 1)
#axes[0].hist(rcat[good]["Z_gal"], bins=60, range=(-100, 100))
#axes[1].hist(vtot[good], bins=60, range=(0, 1000))
#(rcat[sel]["Vx_gal"]/vtot[sel]).max()
#(rcat[sel]["Vx_gal"]/vtot[sel]).mean()

# --- Quiver ---
projections = ["xz", "xy"]
fig, axes = pl.subplots(2, 2, sharex=True, sharey="row")
cbl = [lm_quiver(lm[lmsel], lmz[lmsel], vtot=vtot_lm[lmsel], show=s, ax=ax)
       for s, ax in zip(projections, axes[:, 0])]
cbh = [h3_quiver(rcat[sel], feh[sel], vtot=vtot[sel], show=s, ax=ax)
       for s, ax in zip(projections, axes[:, 1])]

#fig.colorbar(cbl[0], ax=axes[:, 0])
#fig.colorbar(cbh[0], ax=axes[:, 1])


axes[0, 0].set_title("LM10")
axes[0, 1].set_title("H3")
[ax.set_xlabel(s[0].upper()) for s, ax in zip(projections, axes[:, 0])]
[ax.set_xlabel(s[0].upper()) for s, ax in zip(projections, axes[:, 1])]
[ax.set_ylabel(s[1].upper()) for s, ax in zip(projections, axes[:, 0])]
[ax.set_ylabel(s[1].upper()) for s, ax in zip(projections, axes[:, 1])]

fig.subplots_adjust(hspace=0.1, wspace=0.1)

pl.show()

import sys
sys.exit()

# --- W and w/out SGR on-sky---
print(good.sum(), (good & ~sgr).sum())

sfig, saxes = pl.subplots(1, 2, sharex=True, sharey=True)

sel = good & sgr
cb = h3_quiver(rcat, sel, feh, ax=saxes[0])

sel = good & ~sgr
cb = h3_quiver(rcat, sel, feh, ax=saxes[1])

sfig.colorbar(cb, ax=saxes)

# --- Above and Below ---
zfig, zaxes = pl.subplots(1, 2, sharex=True, sharey=True)

sel = good & (rcat["Z_gal"] < 0)
cb = h3_quiver(rcat, sel, feh, ax=zaxes[0])

sel = good & (rcat["Z_gal"] > 0)
cb = h3_quiver(rcat, sel, feh, ax=zaxes[1])

zfig.colorbar(cb, ax=zaxes)

# ----- Vr Vtheta ----

vfig, vax = pl.subplots()
cb = vax.scatter(rcat[sel]["Vr_gal"], rcat["Vtheta_gal"][sel], 
                 c=rcat["R_gal"][sel], marker="o", alpha=0.5, 
                 vmin=0, vmax=100, cmap="viridis")
vax.set_xlim(-300, 300)
vax.set_ylim(-400, 400)
vfig.colorbar(cb)