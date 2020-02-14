#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from astropy import units as u

from .quantities import gsr_to_rv, rv_to_gsr, compute_lstar, sgr_coords


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
                   ("lx",),
                   ("ly",),
                   ("lz",),
                   ("vgsr",),
                   ("etot",)]

lm10_cols = {"ra": "ra", "dec": "dec",
             "pmra": "mua", "pmdec": "mud",
             "dist": "dist",
             "vgsr": "vgsr"}

dl17_cols = {"ra": "ra", "dec": "dec",
             "pmra": "pm_ra", "pmdec": "pm_dec",
             "dist": "dist",
             "vgsr": "vlos"}

r18_cols = {"ra": "ra", "dec": "dec",
            "pmra": "pm_ra", "pmdec": "pm_dec",
            "dist": ("parallax", lambda x: 1/x),
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


def homogenize(cat, catname=""):
    """Construct an auxiliary, row matched, catalog that has a standardized
    set of column nmes for phase spece infomation.
    """
    cmap = COLMAPS[catname]
    #cols = list(cat.dtype.names)
    newcols = [r[0] for r in required_columns + derived_columns]
    dtype = np.dtype([(c, np.float32) for c in newcols])

    ncat = np.zeros(len(cat), dtype=dtype)
    for c, mapping in cmap.items():
        if type(mapping) is tuple:
            ncat[c] = mapping[1](cat[mapping[0]])
        else:
            ncat[c] = cat[mapping]
    return ncat


def rectify(ncat, gc_frame):
    
    # convert LOS velocities
    if ncat["vrad"].max() == 0:
        ncat["vrad"] = gsr_to_rv(ncat, gc_frame=gc_frame).value
    else:
        ncat["vgsr"] = rv_to_gsr(ncat, gc_frame=gc_frame)
    
    # kinematic data
    lstar, gc = compute_lstar(ncat, gc_frame)
    for i, a in enumerate("xyz"):
        ncat["{}_gal".format(a)] = getattr(gc, a).to("kpc").value
        ncat["v{}_gal".format(a)] = getattr(gc, "v_{}".format(a)).to("km/s").value
        ncat["l{}".format(a)] = lstar[:, i] / 1e4
        
    # Sgr coordinates
    sgr = sgr_coords(ncat)
    # wrap the negative coords
    lam = sgr.Lambda.value
    lam[lam < 0] +=360 
    ncat["lambda"] = lam
    ncat["beta"] = sgr.Beta.value

    return ncat