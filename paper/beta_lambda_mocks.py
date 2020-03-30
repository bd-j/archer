#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.plummer import convert_estar_rmax


def show_dlam(cat_r, show, nshow=None, ax=None, colorby=None, randomize=True,
              **scatter_kwargs):
    if randomize:
        if nshow is None:
            nshow = show.sum()
        rand = np.random.choice(show.sum(), size=nshow, replace=False)
    else:
        rand = slice(None)
    rgal = np.sqrt(cat_r["x_gal"]**2 + cat_r["y_gal"]**2 + cat_r["z_gal"]**2)
    #rgal = cat_r["dist"]
    y = cat_r[show][rand]["beta"] + np.random.uniform(-2, 2, size=rand.shape)
    x = cat_r[show][rand]["lambda"] + np.random.uniform(-2, 2, size=rand.shape)
    cb = ax.scatter(x, y,#rgal[show][rand],
                    c=colorby[show][rand], **scatter_kwargs)
    return ax, cb


if __name__ == "__main__":

    zcut = -1.9
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
    rmax, energy = convert_estar_rmax(lm10["estar"])

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17", pcat=pcat, 
                                fractional_distance_error=frac_err),
                     gc_frame_dl17)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    text = [0.9, 0.1]
    bbox = dict(facecolor='white')
    zmin, zmax = -2.0, -0.1
    ncol = 3
    figsize = (11, 9.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(ncol, 1, height_ratios=ncol * [10],
                  left=0.1, right=0.87, hspace=0.2, top=0.93)
    gsc = GridSpec(ncol, 1, left=0.89, right=0.9, hspace=0.2, top=0.93)
                   #bottom=0.89, top=0.95)
    vlaxes = []

    # --- plot H3 ----
    vlaxes.append(fig.add_subplot(gs[0, 0]))
    ax = vlaxes[-1]
    show = good & sgr
    nshow = None
    ax, cbh = show_dlam(rcat_r, show, ax=ax, colorby=rcat["feh"],
                        vmin=zmin, vmax=zmax, cmap="magma",
                        marker='o', s=4, alpha=0.5, zorder=2, linewidth=0)
    ax.text(text[0], text[1], "H3 Giants", transform=ax.transAxes, bbox=bbox)
    # highlight low feh
    show = good & sgr & (rcat["FeH"] < zcut)
    ax, cbh = show_dlam(rcat_r, show, ax=ax, colorby=rcat["feh"],
                        vmin=zmin, vmax=zmax, cmap="magma",
                        marker='o', s=9, alpha=1.0, zorder=3, linewidth=0)
                        #label="[Fe/H] < {}".format(zcut))

    # --- LM10 Mocks ---
    ax = fig.add_subplot(gs[1, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    colorby, cname = 0.66*rmax, r"R$_{\rm prog}$" #r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)"
    vmin, vmax = 0.3, 3
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    show = unbound #& (lm10_r["in_h3"] == 1)
    ax, cbl = show_dlam(lm10_r, show, nshow=nshow, ax=ax, colorby=colorby,
                        vmin=vmin, vmax=vmax, cmap="magma_r",
                        marker='o', linewidth=0, alpha=0.5, s=4)
    ax.text(text[0], text[1], "LM10\n(noiseless)", transform=ax.transAxes, bbox=bbox)

    # --- DL17 Mock ---
    ax = fig.add_subplot(gs[2, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    cm = ListedColormap(["tomato", "black"])
    show = (dl17["id"] >= 0) #& (dl17_r["in_h3"] == 1)
    ax, cbd = show_dlam(dl17_r, show, nshow=nshow, ax=ax, colorby=dl17["id"],
                        vmin=0, vmax=1, cmap=cm, #norm=norm,
                        marker='o', linewidth=0, alpha=1.0, s=4)
    ax.text(text[0], text[1], "DL17\n(noiseless)", transform=ax.transAxes, bbox=bbox)

    # prettify
    [ax.set_xlim(-5, 365) for ax in vlaxes]
    [ax.set_ylim(-90, 90) for ax in vlaxes]
    [ax.set_ylabel(r"$B_{\rm Sgr}$ (deg)" ) for ax in vlaxes]
    [ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in vlaxes[-1:]]
    from matplotlib.lines import Line2D
    points = Line2D([], [], linestyle="", color="black",
                    marker="o", markersize=3)
    vlaxes[0].legend([points], ["[Fe/H] < {}".format(zcut)], loc="upper right")

    # ---- Colorbars ----
    cax1 = fig.add_subplot(gsc[1, -1])
    #pl.colorbar(cb, cax=cax, label=r"$t_{unbound}$ (Gyr)")
    pl.colorbar(cbl, cax=cax1, label=cname)
    cax2 = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")
    cax3 = fig.add_subplot(gsc[2, -1])
    pl.colorbar(cbd, cax=cax3, label=r"", ticks=[0.25, 0.75])
    cax3.set_yticklabels(["Stars", "DM"])

    if config.savefig:
        fig.savefig("{}/beta_lambda_mocks.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
