#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from astropy.io import fits

from archer.config import parser
from archer.cornerplot import twodhist
from archer.chains import ellipse_pars, ellipse_artist
from archer.fitting import best_model, sample_posterior

from archer.figuremaker import FigureMaker

def show_lzly(cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat["lz"][show], cat["ly"][show], c=colorby[show],
                       **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat["lz"][show], cat["ly"][show],
                **plot_kwargs)
        return ax

def remnant_L():
    from archer.frames import sgr_fritz18
    gc = sgr_fritz18.transform_to(config.gc_frame)
    xx = np.array([getattr(gc, a).to("kpc").value for a in "xyz"]).T
    p = np.array([getattr(gc, "v_{}".format(a)).to("km/s").value
                  for a in "xyz"]).T
    Lstar = np.cross(xx, p)
    return Lstar / 1e3


class Plotter(FigureMaker):

    def make_axes(self, nrow, ncol, span=[(-9.9, 11.5), (-14, 11)]):
        self.ms = 3
        figsize = (4 * ncol +2, 4 * nrow + 2)
        self.fig = pl.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec
        self.gs = GridSpec(nrow, ncol, width_ratios=ncol * [10],
                           left=0.1, right=0.95, wspace=0.25, top=0.93)
        #gsc = GridSpec(1, 2, left=0.1, right=0.95, wspace=0.3,
        #               bottom=0.89, top=0.95)
        axes = [self.fig.add_subplot(self.gs[iz, iarm]) for iz in range(nrow) for iarm in range(ncol)]
        self.axes = np.array(axes).reshape(nrow, ncol)

        return self.fig, self.axes


if __name__ == "__main__":

    zbins = [(-0.8, -0.1),
             (-1.9, -0.8),
             (-3, -1.9)]
    colorby = None

    try:
        parser.add_argument("--nsigma", type=float, default=2.)
    except:
        pass
    args = parser.parse_args()
    plotter = Plotter(args)
    config = plotter.config

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = plotter.select(config, selector=rcat_select)
    lsgr = remnant_L()
    n_tot = (good & sgr).sum()
    plotter.select_arms()
    arms = [plotter.trail_sel, plotter.lead_sel]

    # --- plot setup ---
    rcParams = plotter.plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
    nrow, ncol = len(zbins), 2
    fig, axes = plotter.make_axes(nrow, ncol, span=span)

    # --- plot H3 ---
    for iz, zrange in enumerate(zbins):
        with np.errstate(invalid="ignore"):
            inz = (plotter.rcat["FeH"] < zrange[1]) & (plotter.rcat["FeH"] >= zrange[0])
        for iarm, inarm in enumerate(arms):
            ax = axes[iz, iarm]
            show = good & sgr & inz & inarm & plotter.cold
            plotter.show_lylz(ax, show, color='black', alpha=1.0, label="Cold")
            show = good & sgr & inz & inarm & (~plotter.cold)
            plotter.show_lylz(ax, show, color="tomato", alpha=1.0, mew=0.8, zorder=10, linewidth=0.8,
                              fillstyle="none", label="Diffuse")
            show = good & (~sgr) & inz & inarm
            plotter.show_lylz(ax, show, markersize=2, mew=0, color='grey', alpha=1.0)

    # --- plot selection line ---
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5, linestyle="--", color="royalblue", linewidth=2) for ax in axes.flat]
    [ax.plot(zz, 0.3 * zz + 2.5, linestyle="--", color="royalblue", linewidth=1) for ax in axes.flat]

    [ax.plot([lsgr[2]], [lsgr[1]], label="Sgr remnant", linestyle="",
             marker="*", markerfacecolor="royalblue", markersize=8, markeredgecolor="k")
     for ax in axes.flat]

    # --- prettify ---
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit) for ax in axes[:, 0]]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit) for ax in axes[-1, :]]
    [ax.set_ylim(*span[1]) for ax in axes.flat]
    [ax.set_xlim(*span[0]) for ax in axes.flat]
    [ax.axvline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in axes.flat]
    [ax.axhline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in axes.flat]
    axes[0, 0].set_title("Trailing")
    axes[0, 1].set_title("Leading")

    for iz, ax in enumerate(axes[:, 0]):
        ax.text(0.08, 0.9, "{} < [Fe/H] < {}".format(*zbins[iz]),
                transform=ax.transAxes)


    if config.savefig:
        fig.savefig("{}/lylz_cuts.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()