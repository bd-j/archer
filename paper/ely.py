#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17


def show_ely(cat_r, cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat_r["ly"][show], cat["E_tot_pot1"][show]/1e6,
                        c=colorby[show], **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat_r["ly"][show], cat["E_tot_pot1"][show]/1e6,
                **plot_kwargs)
        return ax


if __name__ == "__main__":

    # define low metallicity
    zsplit = -1.9
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # selections
    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] > 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)

    # plot setup
    rcParams = plot_defaults(rcParams)
    ms = 2
    figsize = (10., 4.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, width_ratios=[10, 10],
                  left=0.1, right=0.95, wspace=0.25,
                  top=0.93)
    #gsc = GridSpec(1, 2, left=0.1, right=0.95, wspace=0.3,
    #               bottom=0.89, top=0.95)
    laxes = []

    # plot H3 all feh
    laxes.append(fig.add_subplot(gs[0, 0]))
    ax = show_ely(rcat_r, rcat, good, laxes[0], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey', alpha=0.5,
                   label="H3 Giants")
    ax = show_ely(rcat_r, rcat, good & sgr, ax, linestyle="",
                   marker="o", markersize=ms, mew=0, color='black', alpha=1.0,
                   label="Sgr Selected Giants")
    ax.set_title("All metallicities")
    ax.legend(loc="lower left", fontsize=10)
    
    #plot h3 low FeH
    lowz = rcat["FeH"] < zsplit
    laxes.append(fig.add_subplot(gs[0, 1], sharey=laxes[0], sharex=laxes[0]))
    ax = show_ely(rcat_r, rcat, good & lowz, laxes[-1], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey', alpha=0.5)
    ax = show_ely(rcat_r, rcat, good & sgr & lowz, ax, linestyle="",
                   marker="o", markersize=ms, mew=0, color='black', alpha=1.0,)
    ax.set_title("[Fe/H] < {}".format(zsplit))


    # prettify
    lunit = r" ($10^4 \, {\rm kpc} \, {\rm km} \, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"E$_{\rm tot}$ ($10^6 \, {\rm km}^2 \, {\rm s}^{-2}$)")
     for ax in laxes]
    [ax.set_xlabel(r"L$_{\rm y}$" + lunit)
     for ax in laxes]
    [ax.set_ylim(-0.175, 0.05) for ax in laxes]
    [ax.set_xlim(-1.4, 1.4) for ax in laxes]
    #[ax.axvline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]
    #[ax.axhline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]


    if config.savefig:
        fig.savefig("{}/energy_ly.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)