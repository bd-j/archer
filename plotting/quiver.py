#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from archer.config import parser
from archer.plotting import hquiver
from archer.plummer import convert_estar_rmax

from archer.figuremaker import FigureMaker


def get_axes(rcParams, figxper=5, figyper=5, nrow=1, ncol=1, **extras):

    cdims = 0.03, 0.04, 0.1, figxper*0.1
    edges = 0.1, 0.95

    rightbar = ncol == 1
    topbar = ncol > 1

    mdims = dict(left=edges[0], right=edges[1] - cdims[2] * rightbar,
                 bottom=edges[0], top=edges[1] - cdims[2] * topbar,)
    spacing = dict(hspace=0.15, wspace=0.2,
                   height_ratios=nrow*[10], width_ratios=ncol*[10])
    pdict = spacing
    pdict.update(extras)

    from matplotlib.gridspec import GridSpec
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


class Plotter(FigureMaker):

    def make_axes(self, split=False):
        vlaxes, caxes, fig = [], [], []
        if split:
            ncol = nrow = 1
            for i in range(2):
                f, axes, cax = get_axes(rcParams, ncol=1, nrow=1)
                vlaxes += axes
                caxes += cax
                fig += [f]
        else:
            ncol, nrow = 2, 1
            fig, vlaxes, caxes = get_axes(rcParams, ncol=ncol, nrow=nrow)

        self.hax, self.lax = vlaxes
        self.caxes = caxes
        return fig, ncol

    def show_h3(self, ax, galaxes="xz"):
        # Plot h3
        show = self.sgr_sel & self.good_sel #& (rcat["FeH"] < -1.9)
        ax, cb = hquiver(self.rcat_r, show, colorby=self.rcat["FeH"],
                         ax=ax, axes=galaxes, scale=20, #width=2e-3, alpha=0.9,
                         vmin=-2.0, vmax=-0.1, cmap="magma")
        ax.text(self.text[0], self.text[1], "H3", bbox=self.bbox, transform=ax.transAxes)

        return cb

    def show_lm10(self, ax, colorby=None, cname="", galaxes="xz", **qkwargs):
        nshow = (self.sgr_sel & self.good_sel).sum()
        unbound = self.lm10["tub"] > 0
        mag = self.lm10_seds["PS_r"] + 5 * np.log10(self.lm10["dist"])
        with np.errstate(invalid="ignore"):
            bright = (mag > 15) & (mag < 18.5)
        sel = unbound
        if config.mag_cut:
            sel = sel & bright

        show = sel
        ax, cb = hquiver(self.lm10_rn, show, colorby=colorby,
                        ax=ax, axes=galaxes, nshow=nshow*3,
                        alpha=0.3, **qkwargs)
        show = sel & (self.lm10_rn["in_h3"] > 0.)
        ax, cb = hquiver(self.lm10_rn, show, colorby=colorby,
                        ax=ax, axes=galaxes, nshow=nshow,
                        **qkwargs)
        ax.text(self.text[0], self.text[1], "LM10", bbox=self.bbox, transform=ax.transAxes)

        return cb


if __name__ == "__main__":

    np.random.seed(101)

    try:
        parser.add_argument("--split", action="store_true")
        parser.add_argument("--galaxes", type=str, default="xz")
    except:
        pass

    # --- Setup ---
    args = parser.parse_args()
    plotter = Plotter(args)
    config = plotter.config
    rmax, energy = convert_estar_rmax(plotter.lm10["estar"])

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = plotter.select(config, selector=rcat_select)

    # plot setup
    plotter.plot_defaults(rcParams)
    fig, ncol = plotter.make_axes(split=config.split)
    plotter.bbox = dict(facecolor='white')
    plotter.text = [0.1, 0.85]

    # Plot lm10
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    #colorby, cname = lm10["tub"], r"t$_{\rm unbound}$"
    #vmin, vmax = 0, 5
    colorby, cname = 0.66*0.85*rmax, r"$\hat{\rm R}_{\rm prog}$ (kpc)" #r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)"
    qkwargs = dict(colorby=colorby, cname=cname, vmin=0.25, vmax=2.5, cmap="magma_r")

    # Plots
    cbh = plotter.show_h3(plotter.hax, galaxes=config.galaxes)
    cbl = plotter.show_lm10(plotter.lax, galaxes=config.galaxes, **qkwargs)

    # prettify
    axes = [plotter.hax, plotter.lax]
    [ax.set_xlim(-70, 40) for ax in axes]
    [ax.set_ylim(-80, 80) for ax in axes]
    [ax.set_xlabel(r"{}$_{{\rm Gal}}$ (kpc)".format(config.galaxes[0].upper())) for ax in axes]
    [ax.set_ylabel(r"{}$_{{\rm Gal}}$ (kpc)".format(config.galaxes[1].upper())) for ax in axes]
    [ax.text(-8, 0, r"$\odot$", horizontalalignment='center', verticalalignment='center')
     for ax in axes]

    # colorbars
    labels = [r"[Fe/H]", cname]
    if ncol > 1:
        orient = "horizontal"
    else:
        orient = "vertical"
    for j, cb in enumerate([cbh, cbl]):
        cax = plotter.caxes[j]
        cb = pl.colorbar(cb, cax=cax, label=labels[j], orientation=orient)
    if ncol > 1:
        [ax.xaxis.set_ticks_position("top") for ax in plotter.caxes]
        [ax.xaxis.set_label_position("top") for ax in plotter.caxes]

    if config.savefig:
        if config.split:
            for i, n in enumerate(["h3", "lm10"]):
                name = "{}/quiver_{}.{}".format(config.figure_dir, n, config.figure_extension)
                fig[i].savefig(name, dpi=config.figure_dpi)
        else:
            name = "{}/quiver.{}".format(config.figure_dir, config.figure_extension)
            fig.savefig(name, dpi=config.figure_dpi)
    else:
        pl.show()
