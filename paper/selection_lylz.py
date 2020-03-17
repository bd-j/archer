#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17


def show_lzly(cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat["lz"][show], cat["ly"][show], c=colorby[show],
                       **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat["lz"][show], cat["ly"][show],
                **plot_kwargs)
        return ax


if __name__ == "__main__":

    ncol = 3
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17"), gc_frame_dl17)

    # selections
    good = ((rcat["FLAG"] == 0) & (rcat["SNR"] > 3) &
            (rcat["logg"] < 3.5) & (rcat["FeH"] >= -3))
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    ms = 2
    figsize = (4 * ncol + 2, 4.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, ncol, width_ratios=ncol * [10],
                  left=0.1, right=0.95, wspace=0.25, top=0.93)
    #gsc = GridSpec(1, 2, left=0.1, right=0.95, wspace=0.3,
    #               bottom=0.89, top=0.95)
    laxes = []

    # plot H3
    laxes.append(fig.add_subplot(gs[0, 0]))
    ax = show_lzly(rcat_r, good, laxes[0], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey', alpha=0.5)
    ax.set_title("All H3 Giants")
    
    #plot LM10
    laxes.append(fig.add_subplot(gs[0, 1], sharey=laxes[0], sharex=laxes[0]))
    ax = show_lzly(lm10_r, unbound, laxes[-1], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey', alpha=0.5)
    ax.set_title("LM10")

    #plot DL17
    if ncol > 2:
        laxes.append(fig.add_subplot(gs[0, 2], sharey=laxes[0], sharex=laxes[0]))
        ax = show_lzly(dl17_r, dl17["id"]==0, laxes[-1], linestyle="",
                       marker="o", markersize=ms, mew=0, color='tomato', alpha=0.5, label="Stars")
        #ax = show_lzly(dl17_r, dl17["id"]==1, laxes[-1], linestyle="",
        #               marker="o", markersize=ms, mew=0, color='royalblue', alpha=0.5, label="DM")
        ax.set_title("DL17")
        ax.legend()
    # plot selection line
    zz =  np.linspace(-0.9, 1, 100)
    [ax.plot(zz, -0.3 * zz - 0.25, linestyle="--", color="royalblue", linewidth=2) for ax in laxes]

    # prettify
    lunit = r" ($10^4 \, {\rm kpc} \, {\rm km} \, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit) for ax in laxes]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit) for ax in laxes]
    [ax.set_ylim(-1.4, 1.2) for ax in laxes]
    [ax.set_xlim(-0.99, 1.15) for ax in laxes]
    [ax.axvline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]
    [ax.axhline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]


    if config.savefig:
        fig.savefig("{}/selection_lylz.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)