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
from archer.plummer import convert_estar_rmax


def get_axes(rcParams, figxper=6.5, figyper=6.5, nrow=1, ncol=1, **extras):
    
    cdims = 0.03, 0.04, 0.1, figxper*0.1
    edges = 0.1, 0.95

    rightbar = ncol == 1
    topbar = ncol > 1

    mdims = dict(left=edges[0], right=edges[1] - cdims[2] * rightbar,
                 bottom=edges[0], top=edges[1] - cdims[2] * topbar,)
    spacing = dict(hspace=0.15, wspace=0.15,
                   height_ratios=nrow*[10], width_ratios=ncol*[10])
    pdict = spacing
    pdict.update(extras)

    from matplotlib.gridspec import GridSpec
    rcParams = plot_defaults(rcParams)
    figsize = (figxper * ncol + rightbar*cdims[3],
               figyper * nrow + topbar*cdims[3])
    fig = pl.figure(figsize=figsize)
    
    # main
    pdict.update(mdims)
    gs = GridSpec(nrow, ncol, **pdict)
    axes = [fig.add_subplot(gs[i, j]) for i in range (nrow)
            for j in range(ncol)]

    if rightbar:
        cdims = dict(left=mdims["right"]+cdims[0], right=mdims["right"] + cdims[1],
                     bottom=edges[0], top=edges[1])
        pdict.update(cdims)
        gsc = GridSpec(nrow, 1, **pdict)
        caxes = [fig.add_subplot(gsc[i, 0]) for i in range (nrow)]

    elif topbar:
        cdims = dict(left=edges[0], right=edges[1],
                     bottom=mdims["top"]+cdims[0], top=mdims["top"] + cdims[1])
        pdict.update(cdims)
        gsc = GridSpec(1, ncol, **pdict)
        caxes = [fig.add_subplot(gsc[0, i]) for i in range (ncol)]

    return fig, axes, caxes


if __name__ == "__main__":

    np.random.seed(101)

    try:
        parser.add_argument("--split", action="store_true")
        parser.add_argument("--galaxes", type=str, default="xz")
    except:
        pass

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
    rmax, energy = convert_estar_rmax(lm10["Estar"])

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0


    # plot setup
    text = [0.1, 0.85]
    bbox = dict(facecolor='white')
    vlaxes, cbars, caxes, figs = [], [], [], []
    if config.split:
        ncol = nrow = 1
        for i in range(2):
            fig, axes, cax = get_axes(rcParams, ncol=1, nrow=1)
            vlaxes += axes
            caxes += cax
            figs += [fig]
    else:
        ncol, nrow = 2, 1
        fig, vlaxes, caxes = get_axes(rcParams, ncol=ncol, nrow=nrow)

    hax, lax = vlaxes

    # Plot h3
    show = sgr & good #& (rcat["FeH"] < -1.9)
    nshow = show.sum()
    ax, cb = hquiver(rcat_r, show, colorby=rcat["FeH"],
                     ax=hax, axes=galaxes, scale=20, #width=2e-3, alpha=0.9,
                     vmin=-2.0, vmax=-0.1, cmap="magma")
    ax.text(text[0], text[1], "H3", bbox=bbox, transform=ax.transAxes)
    vlaxes.append(ax)
    cbars.append(cb)
    
    # Plot lm10
    colorby, cname = 0.66*0.85*rmax, r"$\hat{\rm R}_{\rm prog}$ (kpc)" #r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)"
    vmin, vmax = 0.25, 2.5
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    #colorby, cname = lm10["tub"], r"t$_{\rm unbound}$"
    #vmin, vmax = 0, 5
    
    show = unbound
    ax, cb = hquiver(lm10_r, show, colorby=colorby,
                     ax=lax, axes=galaxes, nshow=nshow*3,
                     vmin=vmin, vmax=vmax, cmap="magma_r", alpha=0.3)
    show = unbound & (lm10_r["in_h3"] > 0.)
    ax, cb = hquiver(lm10_r, show, colorby=colorby,
                     ax=ax, axes=galaxes, nshow=nshow,
                     vmin=vmin, vmax=vmax, cmap="magma_r")
    ax.text(text[0], text[1], "LM10", bbox=bbox, transform=ax.transAxes)
    vlaxes.append(ax)
    cbars.append(cb)

    # prettify
    [ax.set_xlim(-70, 40) for ax in vlaxes]
    [ax.set_ylim(-80, 80) for ax in vlaxes]
    [ax.set_xlabel(r"{}$_{{\rm Gal}}$ (kpc)".format(galaxes[0].upper())) for ax in vlaxes]
    [ax.set_ylabel(r"{}$_{{\rm Gal}}$ (kpc)".format(galaxes[1].upper())) for ax in vlaxes]
    [ax.text(-8, 0, r"$\odot$", horizontalalignment='center', verticalalignment='center')
     for ax in vlaxes]

    # colorbars
    labels = [r"[Fe/H]", cname]
    if ncol > 1:
        orient = "horizontal"
    else:
        orient = "vertical"
    for j, cb in enumerate(cbars):
        cax = caxes[j]
        cb = pl.colorbar(cb, cax=cax, label=labels[j], orientation=orient)

    if ncol > 1:
        [ax.xaxis.set_ticks_position("top") for ax in caxes]
        [ax.xaxis.set_label_position("top") for ax in caxes]

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
