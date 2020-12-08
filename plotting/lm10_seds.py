#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.seds import SEDmaker
from archer.plummer import convert_estar_rmax
from archer.frames import gc_frame_law10


def power_rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)


def make_seds(mini=[], feh=[], loga=None, eep=None, maker=None):

    n_star = len(mini)

    cols = ["mini", "feh", "eep", "loga", "logl",
            "logg", "logt", "feh_surf", "agewt"] + maker.filters
    dt = np.dtype([(c, np.float32) for c in cols])
    cat = np.zeros(n_star, dtype=dt)

    for i in range(n_star):
        if np.mod(i, 5000) == 0:
            print("done {} of {}".format(i, n_star))
        pars = dict(mini=mini[i], feh=feh[i])
        if eep is None:
            pars["loga"] = loga[i]
            pars["eep"] = maker.get_eep(smf=1, **pars)
        else:
            pars["eep"] = eep[i]
        mags, params, _ = maker.get_sed(**pars)
        pars.update(params)
        for j, m in enumerate(maker.filters):
            cat[i][m] = mags[j]
        for k, v in pars.items():
            cat[i][k] = v
    return cat


if __name__ == "__main__":

    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype, gaia_vers=config.gaia_vers), config.gc_frame)
    pcat = fits.getdata(config.pcat_file)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)
    rmax, energy = convert_estar_rmax(lm10["estar"])

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0

    maker = SEDmaker(nnfile=config.nnfile, mistfile=config.mistfile,
                     ageweight=False)


    n_star = len(energy)
    feh = -2 * (1 + energy)
    mini = power_rndm(0.76, 0.9, -2.3, size=n_star)
    loga = np.random.normal(10.07, 0.02, size=n_star)
    eep = np.random.uniform(475, 808, size=n_star)

    cat = make_seds(mini=mini, eep=eep, feh=feh, maker=maker)

    outname = os.path.join(os.path.dirname(config.lm10_file), "LM10_seds.fits")
    fits.writeto(outname, cat)

    sys.exit()
    ax.hist(cat["feh"][unbound], bins=20, density=True, alpha=0.5,
            label=r"$0.76 < M_{i} < 0.9, \, \log({\rm Age}) \sim \mathcal{N}(10.07, 0.01)$")
    mag = cat["PS_r"] + 5 * np.log10(lm10_r["dist"])
    sel = (mag < 18.5) & (mag > 15) & (unbound)
    ax.hist(cat["feh"][sel], bins=20, density=True, alpha=0.5,
            label=r"$15 < r < 18.5$")
