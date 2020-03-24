#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10
from archer.plotting import hquiver


if __name__ == "__main__":

    galaxes = "xz"
    np.random.seed(101)

    config = rectify_config(parser.parse_args())
    frac_err = config.fractional_distance_error

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
    pcat = fits.getdata(config.pcat_file)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10", pcat=pcat, 
                                fractional_distance_error=frac_err), 
                     gc_frame_law10)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0


    # plot setup
    rcParams = plot_defaults(rcParams)
    text = [0.1, 0.85]
    bbox = dict(facecolor='white')
    nrow = 2
    ncol = 1
    figsize = (8, 13.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, ncol, height_ratios=nrow * [10],
                  hspace=0.1, wspace=0.08,
                  left=0.12, right=0.86, top=0.95, bottom=0.08)
    gsc = GridSpec(nrow, 1, hspace=0.12,
                   left=0.89, right=0.9, top=0.95, bottom=0.08)
                   #bottom=0.89, top=0.95)
    vlaxes, cbars = [], []

    # Plot h3
    ax = fig.add_subplot(gs[0,0])
    show = sgr & good
    nshow = show.sum()
    ax, cb = hquiver(rcat_r, show, colorby=rcat["FeH"],
                     ax=ax, axes=galaxes,
                     vmin=-2.0, vmax=-0.1, cmap="magma")
    ax.text(text[0], text[1], "H3", bbox=bbox, transform=ax.transAxes)
    vlaxes.append(ax)
    cbars.append(cb)
    
    # Plot lm10
    ax = fig.add_subplot(gs[1, 0])
    show = unbound
    ax, cb = hquiver(lm10_r, show, colorby=lm10["estar"],
                     ax=ax, axes=galaxes, nshow=nshow*2,
                     vmin=0.0, vmax=1.0, cmap="magma", alpha=0.3)
    show = unbound & (lm10_r["in_h3"] > 0.)
    ax, cb = hquiver(lm10_r, show, nshow=nshow, colorby=lm10["estar"],
                     ax=ax, axes=galaxes,
                     vmin=0.0, vmax=1.0, cmap="magma_r")
    ax.text(text[0], text[1], "LM10", bbox=bbox, transform=ax.transAxes)
    vlaxes.append(ax)
    cbars.append(cb)

    # prettify
    [ax.set_xlim(-70, 40) for ax in vlaxes]
    [ax.set_ylim(-80, 80) for ax in vlaxes]
    [ax.set_xlabel(r"{}$_{{\rm Gal}}$ (kpc)".format(galaxes[0].upper())) for ax in vlaxes]
    [ax.set_ylabel(r"{}$_{{\rm Gal}}$ (kpc)".format(galaxes[1].upper())) for ax in vlaxes]

    # colorbars
    labels = [r"[Fe/H]", r"E"]
    for j, cb in enumerate(cbars):
        cax = fig.add_subplot(gsc[j, 0])
        cb = pl.colorbar(cb, cax=cax, label=labels[j])


    if config.savefig:
        fig.savefig("{}/quiver.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()
