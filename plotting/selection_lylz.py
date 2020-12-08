#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from archer.config import parser
from archer.cornerplot import twodhist
from archer.chains import ellipse_pars, ellipse_artist

from archer.figuremaker import FigureMaker


def show_ellipses(rcat, rcat_r, ax=None, covdir="",
                  edgecolor="k", linewidth=0.5,
                  alpha=1.0, facecolor="none"):
    px, rpx, xs = "Lz", "lz", 1e3
    py, rpy, ys = "Ly", "ly", 1e3

    for i, row in enumerate(rcat_r):
        x, y = row[rpx], row[rpy]
        n = rcat[i]["starname"]
        try:
            cxx, cyy, cxy = ellipse_pars(px, py, n, covdir=covdir)
        except(IOError):
            print("Couldn't find {}".format(n))
            continue
        ell = ellipse_artist(x, y, cxx/xs**2, cyy/ys**2, cxy/(xs * ys))
        ell.set_facecolor(facecolor)
        ell.set_edgecolor(edgecolor)
        ell.set_linewidth(linewidth)
        ell.set_alpha(alpha)
        ax.add_artist(ell)


def show_lzly(cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat["lz"][show], cat["ly"][show], c=colorby[show],
                       **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat["lz"][show], cat["ly"][show],
                **plot_kwargs)
        return ax


def remnant_L(gc_frame):
    from archer.frames import sgr_fritz18
    gc = sgr_fritz18.transform_to(gc_frame)
    xx = np.array([getattr(gc, a).to("kpc").value for a in "xyz"]).T
    p = np.array([getattr(gc, "v_{}".format(a)).to("km/s").value
                  for a in "xyz"]).T
    Lstar = np.cross(xx, p)
    return Lstar / 1e3


class Plotter(FigureMaker):

    def make_axes(self, ncol=3):

        self.ms = 2
        self.art = {}
        figsize = (4 * ncol + 2, 4.0)
        self.fig = pl.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec
        self.gs = GridSpec(1, ncol, width_ratios=ncol * [10],
                           left=0.06, right=0.98, wspace=0.25, top=0.93, bottom=0.13)
        self.rax = self.fig.add_subplot(self.gs[0, 0])
        self.lax = self.fig.add_subplot(self.gs[0, 1], sharey=self.rax, sharex=self.rax)
        if ncol > 2:
            self.dax = self.fig.add_subplot(self.gs[0, 2], sharey=self.rax, sharex=self.rax)

    def make_legend(self, ax, art):
        leg = list(art.keys())
        ax.legend([art[l] for l in leg], leg, fontsize=10)

    def show_remnant(self, axes):

        lsgr = remnant_L(self.config.gc_frame)
        [ax.plot([lsgr[2]], [lsgr[1]], label="Sgr remnant", linestyle="",
                 marker="*", markerfacecolor="gold", markersize=12, markeredgecolor="k",
                 ) for ax in axes]

        art = {"Sgr remnant (F18)": Line2D([], [], marker="*", ms=10, markerfacecolor="gold",
                                            markeredgecolor="k", linestyle="")}
        return art

    def plot_rcat(self, ax):
        """Plot Ly-Lz for the RCAT.
        """
        ax.set_title("All H3 Giants")

        good, sgr = self.good_sel, self.sgr_sel
        ax = show_lzly(self.rcat_r, good, ax, linestyle="",
                       marker="o", markersize=self.ms, mew=0, color='black', alpha=0.5)
        if self.config.show_errors:
            show_ellipses(self.rcat[good & sgr], self.rcat_r[good & sgr], ax=ax,
                          covdir=self.config.covar_dir, alpha=0.3)

        return {}


    def plot_lm10(self, ax, span=[(-9.9, 11.5), (-14, 11)]):
        """Plot Ly-Lz for LM10.
        """
        ax.set_title("Law & Majewski (2010)")

        unbound = self.lm10["tub"] > 0
        # --- in H3 ---
        mag = self.lm10_seds["PS_r"] + 5 * np.log10(self.lm10_r["dist"])
        with np.errstate(invalid="ignore"):
            bright = (mag > 15) & (mag < 18.5)
        show = unbound & (self.lm10_rn["in_h3"] == 1)
        if self.config.mag_cut:
            show = show & bright
        ax = show_lzly(self.lm10_rn, show, ax, linestyle="",
                       marker="o", markersize=self.ms, mew=0, color='grey',
                       alpha=0.5, zorder=0)
        # --- all unbound ---
        show = unbound
        _ = twodhist(self.lm10_rn["lz"][show], self.lm10_rn["ly"][show], ax=ax,
                     span=span, fill_contours=False, color="black",
                     contour_kwargs={"linewidths": 0.75})

        art = {"Unbound particles": Line2D([], [], color="k", linewidth=0.75),
               "Within H3 window": Line2D([], [], marker="o", markersize=self.ms, linewidth=0, color="grey")}

        return art


    def plot_dl17(self, ax, span=[(-9.9, 11.5), (-14, 11)]):
        """Plot Ly-Lz for DL17.
        """
        ax.set_title("Dierickx & Loeb (2017)")

        with np.errstate(invalid="ignore"):
            dl_remnant = ((self.dl17_r["ra"] < 315) & (self.dl17_r["ra"] > 285)  &
                          (self.dl17_r["dec"] < -25) & (self.dl17_r["dec"] > -32))

        stars = (self.dl17["id"] < 1) & ~dl_remnant
        _ = twodhist(self.dl17_r["lz"][stars], self.dl17_r["ly"][stars], ax=ax,
                     span=span, fill_contours=True, color="tomato",
                     contour_kwargs={"linewidths": 1.0})
        dark = (self.dl17["id"] == 1) & ~dl_remnant
        _ = twodhist(self.dl17_r["lz"][dark], self.dl17_r["ly"][dark], ax=ax,
                     span=span, fill_contours=False, color="k",
                     contour_kwargs={"linewidths": 0.75})
        ax.plot([np.nanmedian(self.dl17_r["lz"][dl_remnant])], [np.nanmedian(self.dl17_r["ly"][dl_remnant])],
                marker="*", markersize=7, linewidth=0, color="k", label="DL17 remnant")

        art = {"Unbound stars": Line2D([], [], color="tomato"),
               "Dark Matter": Line2D([], [], color="k", linewidth=0.75),
               "Remnant": Line2D([], [], marker="*", markersize=8, linewidth=0, color="k")}

        return art


if __name__ == "__main__":

    try:
        parser.add_argument("--show_errors", action="store_true")
        #parser.add_argument("--show_gcs", action="store_true")
        parser.add_argument("--ncol", type=int, default=2)
        parser.add_argument("--mag_cut", action="store_true")
    except:
        pass

    # --- Setup ---
    args = parser.parse_args()
    plotter = Plotter(args)
    rcParams = plotter.plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
    config = plotter.config

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = plotter.select(plotter.config, selector=rcat_select)

    # --- Make plots ---
    plotter.make_axes(ncol=args.ncol)
    rart = plotter.plot_rcat(plotter.rax)
    lart = plotter.plot_lm10(plotter.lax, span=span)
    axes = [plotter.rax, plotter.lax]
    if args.ncol > 2:
        dart = plotter.plot_dl17(plotter.dax, span=span)
        axes += [plotter.dax]
    sart = plotter.show_remnant(axes)
    plotter.make_legend(plotter.rax, sart)
    plotter.make_legend(plotter.lax, lart)
    plotter.make_legend(plotter.dax, dart)

    # --- plot selection line ---
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5 + config.dly, linestyle="--", color="royalblue", linewidth=2)
     for ax in axes]

    # --- prettify ---
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit, fontsize=14) for ax in axes]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit, fontsize=14) for ax in axes]
    [ax.set_ylim(*span[1]) for ax in axes]
    [ax.set_xlim(*span[0]) for ax in axes]
    [ax.axvline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in axes]
    [ax.axhline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in axes]


    if plotter.config.savefig:
        plotter.fig.savefig("{}/selection_lylz.{}".format(config.figure_dir, config.figure_extension),
                            dpi=config.figure_dpi)
    else:
        pl.show()