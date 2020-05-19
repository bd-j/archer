#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot the afe vs feh for Sgr stars (and bg giants?) as well as an feh histogram
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.lines import Line2D

from astropy.io import fits

from archer.catalogs import homogenize, rectify
from archer.config import parser, rectify_config, plot_defaults


if __name__ == "__main__":

    zmin, zmax = -3.0, 0.05
    zbins = np.arange(zmin, zmax, 0.1)
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)
    wcat = fits.getdata(config.rcat_file.replace("rcat", "wcat"))

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly, flx=config.flx)
    weights = 1./wcat["total_weight"]
    weights[~np.isfinite(weights)] = 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    ms = 2
    figsize = (5, 6.6)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, height_ratios=[10, 10],
                  left=0.13, right=0.95, wspace=0.25, hspace=0.2, top=0.95)
    
    hax = fig.add_subplot(gs[0, 0])
    zax = fig.add_subplot(gs[1, 0])
    
    # --- Plot histogram of feh values ---
    show = good & sgr & (rcat["BHB"] == 0) #& (rcat["XFIT_RANK"] < 3)
    renorm = show.sum() / weights[show].sum()
    wght = weights * renorm
    #wght = np.clip(weights * renorm, 0, 5)
    #wght *= show.sum() / weights[show].sum()

    hax.hist(rcat[show]["feh"], bins=zbins, histtype="step",
             density=False, color="black", linestyle=":", linewidth=2)
    hax.hist(rcat[show]["feh"], weights=wght[show], bins=zbins, histtype="step",
             density=False, color="maroon", linewidth=2)
    #show = show & (rcat["MGIANT"] == 0)
    #renorm = show.sum() / weights[show].sum()
    #hax.hist(rcat[show]["feh"], weights=weights[show]* renorm, bins=zbins, histtype="step",
    #         density=False, color="darkslateblue")
    
    art = {"Raw counts": Line2D([], [], color="black", linestyle=":", linewidth=2),
           "Reweighted": Line2D([], [], color="maroon", linewidth=2),
          }
    leg = list(art.keys())
    hax.legend([art[l] for l in leg], leg, fontsize=12, loc="upper left")
 
    hax.set_xlabel("[Fe/H]")
    hax.set_ylabel("N")
    hax.set_xlim(zmin, zmax)

    # --- Plot afe vs feh ---
    from make_selection import good_select
    show = good_select(rcat, extras=False)
    zax.plot(rcat[show]["feh"], rcat[show]["afe"],
             marker="o", mew=0, linewidth=0, markersize=1,
             color="grey", alpha=0.3, linestyle="", label="All Giants, S/N>5")
    
    show = good & sgr & (rcat["BHB"] == 0) & (rcat["SNR"] > 5)
    zax.plot(rcat[show]["feh"], rcat[show]["afe"],
             marker="o", mew=0, linewidth=0, markersize=2,
             color="black", linestyle="", label="Sgr, S/N>5")
    zax.set_xlabel("[Fe/H]")
    zax.set_ylabel(r"[$\alpha$/Fe]")
    zax.set_xlim(zmin, zmax)
    zax.set_ylim(-0.3, 0.7)

    zax.errorbar([-2.7], [-0.1], xerr=0.1, yerr=0.1, capsize=4, alpha=0.8, color="grey")
    zz = np.linspace(-2, 0.0, 20)
    zax.plot(zz, -0.33*zz + 0.2, color='k', linestyle='--', linewidth=1, alpha=0.4)

    #zax.legend(fontsize=10, loc="upper right")

    if config.savefig:
        fig.savefig("{}/metallicities.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
