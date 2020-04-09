#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.cornerplot import twodhist
from archer.chains import ellipse_pars, ellipse_artist


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


if __name__ == "__main__":

    ncol = 2
    try:
        parser.add_argument("--show_errors", action="store_true")
    except:
        pass
    config = rectify_config(parser.parse_args())
    frac_err = config.fractional_distance_error
    pcat = fits.getdata(config.pcat_file)

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)

    # noisy lm10
    lm10_rn = rectify(homogenize(lm10, "LM10", pcat=pcat, 
                                 fractional_distance_error=frac_err), 
                      gc_frame_law10)

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17"), gc_frame_dl17)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
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
                   marker="o", markersize=ms, mew=0, color='black', alpha=0.5)
    if config.show_errors:
        show_ellipses(rcat[good & sgr], rcat_r[good & sgr], ax=ax, 
                      covdir=config.covar_dir, alpha=0.3)

    ax.set_title("All H3 Giants")
    
    #plot LM10
    laxes.append(fig.add_subplot(gs[0, 1], sharey=laxes[0], sharex=laxes[0]))
    show = unbound & (lm10_rn["in_h3"] == 1)
    ax = show_lzly(lm10_rn, show, laxes[-1], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey',
                   alpha=0.5, zorder=0)
    show = unbound
    _ = twodhist(lm10_r["lz"][show], lm10_r["ly"][show], ax=laxes[-1],
                  span=span, fill_contours=False, color="black",
                  contour_kwargs={"linewidths": 0.75})
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
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5, linestyle="--", color="royalblue", linewidth=2) for ax in laxes]

    # prettify
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit) for ax in laxes]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit) for ax in laxes]
    [ax.set_ylim(-14, 11) for ax in laxes]
    [ax.set_xlim(-9.9, 11.5) for ax in laxes]
    [ax.axvline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in laxes]
    [ax.axhline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in laxes]


    if config.savefig:
        fig.savefig("{}/selection_lylz.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()