import numpy as np
from astropy.io import fits

__all__ = ["h3_quiver", "lm_quiver", "read_lm", "get_sgr"]


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


def read_lm(lmfile):

    try:
        lm = fits.getdata(lmockfile)
        # switch colums
        for a in "xyz":
            lm["{}gc".format(a)] = lm["{}_gal".format(a.upper())]
            lm[vmap[a]] = lm["V{}_gal".format(a)]

    except(OSError):
        with open(lmfile, "r") as f:
            cols = f.readline().split()
            dt = np.dtype([(c, np.float) for c in cols])
            lm = np.genfromtxt(lmfile, skip_header=1, dtype=dt)
        # Make right handed
        lm["xgc"] = -lm["xgc"]
        lm["u"] = -lm["u"]
    
    return lm


def get_sgr(cat):
    import astropy.units as u
    import astropy.coordinates as coord
    import gala.coordinates as gc
    ceq = coord.ICRS(ra=cat['GaiaDR2_ra']*u.deg, dec=cat['GaiaDR2_dec']*u.deg,
                     distance=cat["dist_MS"] * u.kpc,
                     pm_ra_cosdec=cat['GaiaDR2_pmra']*u.mas/u.yr, pm_dec=cat['GaiaDR2_pmdec']*u.mas/u.yr,
                     radial_velocity=cat['Vrad']*u.km/u.s)
    sgr = ceq.transform_to(gc.Sagittarius)
    return sgr
