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


def show_vlam():
    pass


if __name__ == "__main__":

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
    nshow = show.sum()
    cbh = ax.scatter(rcat_r[show]["lambda"], rcat_r[show]["vgsr"],
                     c=rcat[show]["feh"], vmin=-2.5, vmax=0, cmap="magma",
                     marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
    ax.text(text[0], text[1], "H3 Giants", transform=ax.transAxes, bbox=bbox)

    # --- LM10 Mocks ---
    ax = fig.add_subplot(gs[1, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    show = unbound & (lm10_r["in_h3"] == 1)
    rand = np.random.choice(show.sum(), size=nshow, replace=False)
    cbl = ax.scatter(lm10_r[show][rand]["lambda"], lm10_r[show][rand]["vgsr"],
                     c=lm10[show][rand]["Estar"], cmap="rainbow_r",
                     #marker='+', linewidth=1, alpha=0.5, vmin=tmin, vmax=8., s=9,
                     marker='o', linewidth=0, alpha=1.0, vmin=0, vmax=1., s=4)
    ax.text(text[0], text[1], "LM10", transform=ax.transAxes, bbox=bbox)

    # --- DL17 Mock ---
    ax = fig.add_subplot(gs[2, 0], sharey=vlaxes[0], sharex=vlaxes[0])
    vlaxes.append(ax)
    cm = ListedColormap(["tomato", "royalblue"])
    show = (dl17["id"] >= 0) & (dl17_r["in_h3"] == 1)
    rand = np.random.choice(show.sum(), size=nshow, replace=False)
    #norm = BoundaryNorm([-1, 0.5, 5], cm.N)
    cbd = ax.scatter(dl17_r[show][rand]["lambda"], dl17_r[show][rand]["vgsr"],
                     c=dl17[show][rand]["id"], cmap=cm, #norm=norm,
                     marker='o', linewidth=0, alpha=1.0, vmin=0, vmax=1, s=4)
 
    ax.text(text[0], text[1], "DL17", transform=ax.transAxes, bbox=bbox)

    # prettify
    [ax.set_xlim(-5, 365) for ax in vlaxes]
    [ax.set_ylim(-330, 330) for ax in vlaxes]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \cdot {\rm s}^{-1}$)") for ax in vlaxes]
    [ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in vlaxes[-1:]]

    # ---- Colorbars ----
    cax1 = fig.add_subplot(gsc[1, -1])
    #pl.colorbar(cb, cax=cax, label=r"$t_{unbound}$ (Gyr)")
    pl.colorbar(cbl, cax=cax1, label=r"$E_*$")
    cax2 = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")
    cax3 = fig.add_subplot(gsc[2, -1])
    pl.colorbar(cbd, cax=cax3, label=r"", ticks=[0.25, 0.75])
    cax3.set_yticklabels(["Stars", "DM"])

    if config.savefig:
        fig.savefig("{}/vgsr_lambda_mocksXh3.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
