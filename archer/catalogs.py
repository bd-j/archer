#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from astropy import units as u

from .quantities import compute_lstar, sgr_coords
from .quantities import rv_to_gsr, gsr_to_rv, reflex_uncorrect
from .frames import gc_frame_dl17


required_columns = [("ra", u.deg),
                    ("dec", u.deg),
                    ("pmra", u.mas / u.yr),
                    ("pmdec", u.mas / u.yr),
                    ("dist", u.kpc, "heliocentric"),
                    ("vrad", u.km / u.s, "heliocentric"),
                    ("flag", int)
                    ]

derived_columns = [("x_gal", u.kpc, "galactocentric"),
                   ("y_gal", u.kpc, "galactocentric"),
                   ("z_gal", u.kpc, "galactocentric"),
                   ("vx_gal", u.km / u.s, "galactocentric"),
                   ("vy_gal", u.km / u.s, "galactocentric"),
                   ("vz_gal", u.km / u.s, "galactocentric"),
                   ("lambda", u.deg, "LM10 system"),
                   ("beta", u.deg, "LM10 system"),
                   ("lx", u.kpc * u.km / u.s),
                   ("ly", u.kpc * u.km / u.s),
                   ("lz", u.kpc * u.km / u.s),
                   ("vgsr", u.km / u.s, "Galactic standard of rest"),
                   ("etot", (u.km / u.s)**2),
                   ("in_h3", bool)]

lm10_cols = {"ra": "ra", "dec": "dec",
             "pmra": "mua", "pmdec": "mud",
             "dist": "dist",
             "vgsr": "vgsr"}

dl17_cols = {"ra": "ra", "dec": "dec",
             # Note the pms are solar reflex corrected
             "pmra": "pmRA", "pmdec": "pmdec",
             "dist": "dist",
             # this is solar reflex corrected (i.e. is vgsr)
             "vrad": "vlos"}

r18_cols = {"ra": "ra", "dec": "dec",
            "pmra": "pm_ra", "pmdec": "pm_dec",
            "dist": ("parallax", lambda x: 1 / x),
            "vrad": "radial_velocity"}

kcat_cols = {"ra": "ra", "dec": "dec",
             "pmra": "gaia.pmra", "pmdec": "gaia.pmdec",
             "dist": "Dist",
             "vrad": "HRV"}

rcat_cols = {"ra": "RA", "dec": "DEC",
             "pmra": "GAIADR2_PMRA", "pmdec": "GAIADR2_PMDEC",
             "dist": "dist_adpt",
             "vrad": "Vrad",
             "flag": "FLAG"}


# Map the required column names
COLMAPS = {"LM10": lm10_cols,
           "DL17": dl17_cols,
           "R18": r18_cols,
           "KSEGUE": kcat_cols,
           "RCAT": rcat_cols}


def homogenize(cat, catname="", pcat=None,
               fractional_distance_error=0.0):
    """Construct an auxiliary, row matched, catalog that has a standardized
    set of column names for phase spece infomation.

    Parameters
    ----------
    cat : numpy structured array

    catname : str
        One of "RCAT", "DL17", "LM10", "KSEGUE", or "R18"

    fractional_distance_error: float, optional (default 0.0)
        Fractional distance error to add to heliocentric distances
    """
    try:
        cmap = COLMAPS[catname]
    except(KeyError):
        raise KeyError("`catname` must be one of {}".format(COLMAPS.keys()))
    #cols = list(cat.dtype.names)
    newcols = [r[0] for r in required_columns + derived_columns]
    dtype = np.dtype([(c, np.float32) for c in newcols])

    # Copy columns over
    ncat = np.zeros(len(cat), dtype=dtype)
    for c, mapping in cmap.items():
        if type(mapping) is tuple:
            ncat[c] = mapping[1](cat[mapping[0]])
        else:
            ncat[c] = cat[mapping]

    if catname == "DL17":
        # reflex uncorrect the DL17 values
        ncat = reflex_uncorrect(cat=ncat, gc_frame=gc_frame_dl17)
        # id stars in the progenitor?

    if fractional_distance_error > 0.0:
        # Noise up the mocks
        ncat["dist"] *= np.random.normal(1.0, fractional_distance_error,
                                         size=len(ncat))

    if pcat is not None:
        ncat["in_h3"][:] = in_h3(ncat, pcat)

    return ncat


def rectify(ncat, gc_frame):
    """Make sure all columns in auxiliary catalog are correct.

    Parameters
    ----------
    ncat : numpy structured array
        The auxialiary catalog

    gc_frame : astropy.coordinates.Frame
        The galactocentric frame used to convert between GSR and Heliocentric
        velocities.
    """
    # convert LOS velocities
    if ncat["vrad"].max() == 0:
        ncat["vrad"] = gsr_to_rv(ncat, gc_frame=gc_frame).value
    else:
        ncat["vgsr"] = rv_to_gsr(ncat, gc_frame=gc_frame)

    # kinematic data
    lstar, gc = compute_lstar(ncat, gc_frame)
    for i, a in enumerate("xyz"):
        ncat["{}_gal".format(a)] = getattr(gc, a).to("kpc").value
        va = getattr(gc, "v_{}".format(a)).to("km/s").value
        ncat["v{}_gal".format(a)] = va
        ncat["l{}".format(a)] = lstar[:, i] / 1e3

    # Sgr coordinates
    sgr = sgr_coords(ncat)
    # wrap the negative coords
    lam = sgr.Lambda.value
    lam[lam < 0] += 360
    ncat["lambda"] = lam
    ncat["beta"] = sgr.Beta.value

    return ncat


def in_h3(hcat, pcat, radius=3.0):
    """
    Parameters
    ----------
    hcat : structured ndarray
        homogenized (and possibly rectified) catalog

    pcat : structured ndarray
        pointing catalog

    expand : float
        radius to use around the pcat centers, degrees
    """
    from astropy.coordinates import SkyCoord
    pcoord = SkyCoord(pcat['ra_obs'] * u.degree, pcat['dec_obs'] * u.degree)
    hcoord = SkyCoord(hcat['ra'] * u.degree, hcat['dec'] * u.degree)
    _, sep2d, _ = hcoord.match_to_catalog_sky(pcoord)
    sep_mask = (sep2d.degree < radius)
    return sep_mask
