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


def get_axes(rcParams, figxper=8, figyper=13.5/2, nrow=1, ncol=1):
    
    from matplotlib.gridspec import GridSpec
    rcParams = plot_defaults(rcParams)
    figsize = (figxper * ncol, figyper * nrow)
    fig = pl.figure(figsize=figsize)
    gs = GridSpec(nrow, ncol, height_ratios=nrow * [10],
                  hspace=0.1, wspace=0.08,
                  left=0.12, right=0.86, top=0.95, bottom=0.08)
    gsc = GridSpec(nrow, 1, hspace=0.12,
                   left=0.89, right=0.9, top=0.95, bottom=0.08)
            #bottom=0.89, top=0.95)

    axes = [fig.add_subplot(gs[i, 0]) for i in range (nrow)]
    caxes = [fig.add_subplot(gsc[i, 0]) for i in range (nrow)]
    return fig, axes, caxes


if __name__ == "__main__":

    np.random.seed(101)

    parser.add_argument("--split", action="store_true")
    parser.add_argument("--galaxes", type=str, default="xz")
    config = rectify_config(parser.parse_args())
    frac_err = config.fractional_distance_error
    galaxes = config.galaxes

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
    text = [0.1, 0.85]
    bbox = dict(facecolor='white')
    vlaxes, cbars, caxes, figs = [], [], [], []
    if config.split:
        for i in range(2):
            fig, axes, cax = get_axes(rcParams, nrow=1)
            vlaxes += axes
            caxes += cax
            figs += [fig]
    else:
        fig, vlaxes, caxes = get_axes(rcParams, nrow=2)

    hax, lax = vlaxes

    # Plot h3
    show = sgr & good
    nshow = show.sum()
    ax, cb = hquiver(rcat_r, show, colorby=rcat["FeH"],
                     ax=hax, axes=galaxes,
                     vmin=-2.0, vmax=-0.1, cmap="magma")
    ax.text(text[0], text[1], "H3", bbox=bbox, transform=ax.transAxes)
    vlaxes.append(ax)
    cbars.append(cb)
    
    # Plot lm10
    show = unbound
    ax, cb = hquiver(lm10_r, show, colorby=lm10["estar"],
                     ax=lax, axes=galaxes, nshow=nshow*4,
                     vmin=0.0, vmax=1.0, cmap="magma", alpha=0.3)
    show = unbound & (lm10_r["in_h3"] > 0.)
    ax, cb = hquiver(lm10_r, show, colorby=lm10["estar"],
                     ax=ax, axes=galaxes, nshow=nshow,
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
    labels = [r"[Fe/H]", r"E$_\ast$"]
    for j, cb in enumerate(cbars):
        cax = caxes[j]
        cb = pl.colorbar(cb, cax=cax, label=labels[j])


    if config.savefig:
        if config.split:
            for i, n in enumerate(["h3", "lm10"]):
                name = "{}/quiver_{}.{}".format(config.figure_dir, n, config.figure_extension)
                figs[i].savefig(name, dpi=config.figure_dpi)
        else:
            name = "{}/quiver.{}".format(config.figure_dir, config.figure_extension)
            fig.savefig(name, dpi=config.figure_dpi)
    else:
        pl.show()
