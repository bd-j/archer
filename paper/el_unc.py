#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.chains import ellipse_pars, ellipse_artist


parmap = {"Ly": ("ly", 1e3),
          "Lz": ("lz", 1e3),
          "E_tot_pot1": ("E_tot_pot1", 1e5)}


def get(p, cat_r, cat):
    if parmap[p][0] in cat_r.dtype.names:
        return cat_r[parmap[p][0]]
    else:
        return cat[p] / parmap[p][1]
    

def show_ellipses(rcat, rcat_r, ax=None, pars=[], covdir="",
                  edgecolor="k", linewidth=0.5,
                  alpha=1.0, facecolor="none"):
    px, py = pars
    xs, ys = parmap[px][1], parmap[py][1]
    
    rarr = np.array(rcat)
    for i, row in enumerate(rcat_r):
        x = get(px, row, rarr[i])
        y = get(py, row, rarr[i])
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


def show_ely(cat_r, cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat_r["ly"][show], cat["E_tot_pot1"][show]/1e5,
                        c=colorby[show], **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat_r["ly"][show], cat["E_tot_pot1"][show]/1e5,
                **plot_kwargs)
        return ax


def show_lylz(cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat["lz"][show], cat["ly"][show], c=colorby[show],
                       **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat["lz"][show], cat["ly"][show],
                **plot_kwargs)
        return ax


if __name__ == "__main__":

    # define low metallicity
    zsplit = -1.9
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type
    config.show_errors = True

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly, flx=config.flx)
    lowz = rcat["FeH"] < zsplit

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
    
    # --- plot H3 ly-lz ---
    laxes.append(fig.add_subplot(gs[0, 0]))
    ax = show_lylz(rcat_r, good & lowz, laxes[0], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey', alpha=0.5,
                   label="H3 Giants")
    ax = show_lylz(rcat_r, good & sgr & lowz, ax, linestyle="",
                   marker="o", markersize=ms, mew=0, color='black', alpha=1.0,
                   label="[Fe/H]$<${}".format(zsplit))
    if config.show_errors:
        show_ellipses(rcat[good & sgr & lowz], rcat_r[good & sgr & lowz], ax=ax,
                      pars=["Lz", "Ly"], covdir=config.covar_dir, alpha=0.3)
    #ax.set_title("All metallicities")
        # plot selection line
    zz =  np.linspace(-9, 10, 100)
    ax.plot(zz, -0.3 * zz - 2.5 + config.dly, linestyle="--", color="royalblue", linewidth=2, label="Sgr Selection")
    #ax.legend(loc="lower left", fontsize=10)
    ax.set_title("[Fe/H] < {}".format(zsplit))

    # --- plot h3 e-ly ---
    laxes.append(fig.add_subplot(gs[0, 1]))
    ax = show_ely(rcat_r, rcat, good & lowz, laxes[-1], linestyle="",
                  marker="o", markersize=ms, mew=0, color='grey', alpha=0.5)
    ax = show_ely(rcat_r, rcat, good & sgr & lowz, ax, linestyle="",
                  marker="o", markersize=ms, mew=0, color='black', alpha=1.0,)
    if config.show_errors:
        show_ellipses(rcat[good & sgr & lowz], rcat_r[good & sgr & lowz], ax=ax,
                      pars=["Ly", "E_tot_pot1"], covdir=config.covar_dir, alpha=0.3)
    ax.set_title("[Fe/H] < {}".format(zsplit))

    # prettify
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"

    laxes[1].set_ylabel(r"E$_{\rm tot}$ ($10^5 \,\, {\rm km}^2 \,\, {\rm s}^{-2}$)")
    laxes[1].set_xlabel(r"L$_{\rm y}$" + lunit)
    laxes[1].set_ylim(-1.75, 0.25)
    laxes[1].set_xlim(-14, 9)

    laxes[0].set_ylim(-14., 9)
    laxes[0].set_xlim(-8, 6)
    laxes[0].set_ylabel(r"L$_{\rm y}$" + lunit)
    laxes[0].set_xlabel(r"L$_{\rm z}$" + lunit)

    #[ax.axvline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]
    #[ax.axhline(0, linestyle=":", color="k", alpha=0.8) for ax in laxes]

    if config.savefig:
        fig.savefig("{}/l_energy_unc.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()