#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.cornerplot import twodhist
from archer.chains import ellipse_pars, ellipse_artist
from archer.fitting import best_model, sample_posterior


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


if __name__ == "__main__":

    zbins = [(-0.8, -0.1),
             (-1.9, -0.8),
             (-3, -1.9)]
    colorby = None

    try:
        parser.add_argument("--nsigma", type=float, default=2.)
    except:
        pass
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type
    frac_err = config.fractional_distance_error
    nrow = len(zbins)
    ncol = 2
    pcat = fits.getdata(config.pcat_file)

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)
    lsgr = remnant_L()

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = rcat_select(rcat, rcat_r, max_rank=config.max_rank,
                            dly=config.dly, flx=config.flx)
    n_tot = (good & sgr).sum()

    trail = rcat_r["lambda"] < 175
    lead = rcat_r["lambda"] > 175
    arms = [trail, lead]

    # trailing
    tsel = good & sgr & trail
    tmu, tsig, _ = best_model("fits/h3_trailing_fit.h5", rcat_r["lambda"])
    cold_trail = np.abs(rcat_r["vgsr"] - tmu) < (config.nsigma * tsig)
    delta_vt = np.abs(rcat_r["vgsr"] - tmu) / (config.nsigma * tsig)

    # leading
    lsel = good & sgr & lead
    lmu, lsig, _ = best_model("fits/h3_leading_fit.h5", rcat_r["lambda"])
    cold_lead =  np.abs(rcat_r["vgsr"] - lmu) < (config.nsigma * lsig)
    delta_vl = np.abs(rcat_r["vgsr"] - lmu) / (config.nsigma * lsig)

    cold = (lead & cold_lead) | (trail & cold_trail)
    deltas = [delta_vt, delta_vl]

    # --- plot setup ---
    rcParams = plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
    ms = 3
    figsize = (4 * ncol +2, 4 * nrow + 2)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, ncol, width_ratios=ncol * [10],
                  left=0.1, right=0.95, wspace=0.25, top=0.93)
    #gsc = GridSpec(1, 2, left=0.1, right=0.95, wspace=0.3,
    #               bottom=0.89, top=0.95)
    axes = []

    # --- plot H3 ---
    for iz, zrange in enumerate(zbins):
        for iarm, inarm in enumerate(arms):
            ax = fig.add_subplot(gs[iz, iarm])
            with np.errstate(invalid="ignore"):
                inz = (rcat["FeH"] < zrange[1]) & (rcat["FeH"] >= zrange[0])
            show = good & sgr & inz & inarm & cold
            ax = show_lzly(rcat_r, show, ax, linestyle="",
                           marker="o", markersize=ms, mew=0, color='black', alpha=1.0)
            show = good & sgr & inz & inarm & (~cold)
            #ax, cb = show_lzly(rcat_r, show, ax, colorby=deltas[iarm],
            #                   vmin=0, vmax=10, marker='o', s=ms**2, alpha=1.0)
            ax = show_lzly(rcat_r, show, ax,  color="tomato", linestyle="", mew=0.7,
                           marker='o', ms=ms, alpha=1.0, zorder=10, linewidth=0.7, label="Diffuse",
                           fillstyle="none")

            show = good & (~sgr) & inz & inarm
            ax = show_lzly(rcat_r, show, ax, linestyle="",
                           marker="o", markersize=2, mew=0, color='grey', alpha=1.0)

            axes.append(ax)

    # --- plot selection line ---
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5, linestyle="--", color="royalblue", linewidth=2) for ax in axes]
    [ax.plot(zz, 0.3 * zz + 2.5, linestyle="--", color="royalblue", linewidth=1) for ax in axes]

    [ax.plot([lsgr[2]], [lsgr[1]], label="Sgr remnant", linestyle="",
             marker="*", markerfacecolor="royalblue", markersize=8, markeredgecolor="k",
             ) for ax in axes]

    axes = np.array(axes).reshape(nrow, ncol)

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