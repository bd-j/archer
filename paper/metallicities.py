#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the afe vs feh for Sgr stars (and bg giants?) as well as an feh histogram
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.catalogs import homogenize, rectify
from archer.config import parser, rectify_config, plot_defaults


if __name__ == "__main__":

    zmin, zmax = -3.0, 0.05
    zbins = np.arange(zmin, zmax, 0.05)
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # selections
    good = (rcat["FLAG"] == 0) & (rcat["SNR"] > 3) & (rcat["logg"] < 3.5)
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)
    sel = sgr & good

    # plot setup
    rcParams = plot_defaults(rcParams)
    ms = 2
    figsize = (8, 6)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[7, 10],
                  left=0.1, right=0.95, wspace=0.25, hspace=0.2, top=0.95)
    
    hax = fig.add_subplot(gs[0, 0])
    zax = fig.add_subplot(gs[1, 0])
    
    # Plot histogram of feh values
    hax.hist(rcat[sel]["feh"], bins=zbins, histtype="step",
             density=False, color="black")
    hax.set_xlabel("[Fe/H]")
    hax.set_ylabel("N")
    hax.set_xlim(zmin, zmax)

    # Plot afe vs feh
    # zax.errorbar()
    zax.plot(rcat[sel]["feh"], rcat[sel]["afe"],
             marker="o", mew=0, linewidth=0, markersize=2,
             color="black", linestyle="")
    zax.set_xlabel("[Fe/H]")
    zax.set_ylabel(r"[$\alpha$/Fe]")
    zax.set_xlim(zmin, zmax)
    zax.set_ylim(-0.3, 0.7)

    if config.savefig:
        fig.savefig("{}/metallicities.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
