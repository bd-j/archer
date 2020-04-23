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


def show_vlam():
    pass


if __name__ == "__main__":

    zcut = -1.9
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)
    rmax, energy = convert_estar_rmax(lm10["Estar"])

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17"), gc_frame_dl17)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    text = [0.9, 0.1]
    bbox = dict(facecolor='white')
    zmin, zmax = -2, -0.1
    ncol = 3
    figsize = (11, 8.5)
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
    #ax.plot(rcat_r[good & ~sgr]["lambda"], rcat[good & ~sgr]["vgsr"],
    #        'o', markersize=ms, alpha=0.2, color="grey", zorder=1, mew=0)
    show = good & sgr
    cbh = ax.scatter(rcat_r[show]["lambda"], rcat_r[show]["vgsr"],
                     c=rcat[show]["feh"], vmin=zmin, vmax=zmax, cmap="magma",
                     marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
    ax.text(text[0], text[1], "H3 Giants", transform=ax.transAxes, bbox=bbox)
    # highlight low feh
    show = good & sgr & (rcat["FeH"] < zcut)
    cbh = ax.scatter(rcat_r[show]["lambda"], rcat_r[show]["vgsr"],
                     c=rcat[show]["feh"], vmin=zmin, vmax=zmax, cmap="magma",
                     marker='o', s=9, alpha=0.8, zorder=3, linewidth=0,)
                     #label="[Fe/H] < {}".format(zcut))

    # --- LM10 Mocks ---
    ax = fig.add_subplot(gs[1, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    colorby, cname = 0.66*0.85*rmax, r"$\hat{\rm R}_{\rm prog}$ (kpc)" #r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)"
    vmin, vmax = 0.25, 2.5
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    #colorby, cname = lm10["tub"], r"t$_{\rm unbound}$"
    #vmin, vmax = 0, 5
    show = unbound
    rand = np.random.choice(show.sum(), size=show.sum(), replace=False)
    cbl = ax.scatter(lm10_r[show][rand]["lambda"], lm10_r[show][rand]["vgsr"],
                     c=colorby[show][rand], cmap="magma_r",
                     #marker='+', linewidth=1, alpha=0.5, vmin=tmin, vmax=8., s=9,
                     marker='o', linewidth=0, alpha=0.5, vmin=vmin, vmax=vmax, s=2)
    ax.text(text[0], text[1], "LM10\n(noiseless)", transform=ax.transAxes,
            bbox=bbox)

    # --- DL17 Mock ---
    ax = fig.add_subplot(gs[2, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    cm = ListedColormap(["tomato", "black"])
    show = dl17["id"] >= 0
    rand = np.random.choice(show.sum(), size=show.sum(), replace=False)
    #norm = BoundaryNorm([-1, 0.5, 5], cm.N)
    cbd = ax.scatter(dl17_r[show][rand]["lambda"], dl17_r[show][rand]["vgsr"],
                     c=dl17[show][rand]["id"], cmap=cm, #norm=norm,
                     marker='o', linewidth=0, alpha=1.0, vmin=0, vmax=1, s=4)
 
    ax.text(text[0], text[1], "DL17\n(noiseless)", transform=ax.transAxes,
            bbox=bbox)

    # prettify
    [ax.set_xlim(-5, 365) for ax in vlaxes]
    [ax.set_ylim(-330, 330) for ax in vlaxes]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in vlaxes]
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
        fig.savefig("{}/vgsr_lambda_mocks.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
