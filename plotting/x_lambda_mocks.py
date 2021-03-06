#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.quantities import compute_vtan
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.plummer import convert_estar_rmax


def show_xlam(cat_r, show, dist=False, ax=None, colorby=None, randomize=True,
              **scatter_kwargs):
    if randomize:
        rand = np.random.choice(show.sum(), size=show.sum(), replace=False)
    else:
        rand = slice(None)
    x = cat_r[show][rand]["lambda"]
    if dist:
        rgal = np.sqrt(cat_r["x_gal"]**2 + cat_r["y_gal"]**2 + cat_r["z_gal"]**2)
        y = rgal[show][rand]
    else:
        y = cat_r["vgsr"][show][rand]
    if colorby is not None:
        z = colorby[show][rand]
        cb = ax.scatter(x, y, c=z, **scatter_kwargs)
    else:
        cb = ax.plot(x, y, **scatter_kwargs)
    return cb


if __name__ == "__main__":

    try:
        parser.add_argument("--feh_cut", type=float, default=-1.9)
        parser.add_argument("--pcol_limit", type=int, default=8)
        parser.add_argument("--by_vtan", action="store_true")
    except:
        pass
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type
    zcut = config.feh_cut

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype, gaia_vers=config.gaia_vers), config.gc_frame)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)
    rmax, energy = convert_estar_rmax(lm10["Estar"])

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17"), gc_frame_dl17)

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = rcat_select(rcat, rcat_r, max_rank=config.max_rank,
                            dly=config.dly, flx=config.flx)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    text, bbox = [0.05, 0.1], dict(facecolor='white')
    nrow = 3
    figsize = (11, 6.6)  # 14, 8.5
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, 2, height_ratios=nrow * [10],
                  hspace=0.2, wspace=0.15,
                  left=0.08, right=0.9,  bottom=0.08, top=0.93)
    gsc = GridSpec(nrow, 1, left=0.92, right=0.93, hspace=0.2,
                   bottom=0.08, top=0.93)
                   #bottom=0.89, top=0.95)
    axes = np.array([fig.add_subplot(gs[i, j])
                     for i in range(nrow) for j in range(2)]).reshape(nrow, 2)

    # --- plot H3 ----
    colorby = rcat["feh"]
    zmin, zmax = -2, -0.1
    cmap = "magma"
    if config.by_vtan:
        colorby = compute_vtan(rcat_r)
        zmin, zmax = -100, 500
        cmap = "magma_r"

    for i in range(2):
        ax = axes[0, i]
        show = good & sgr
        cbh = show_xlam(rcat_r, show, dist=bool(i), ax=ax, colorby=colorby,
                        vmin=zmin, vmax=zmax, cmap=cmap,
                        marker='o', s=4, alpha=1.0, zorder=2, linewidth=0)
        # highlight low feh
        with np.errstate(invalid="ignore"):
            show = good & sgr & (rcat["FeH"] < zcut)
        cbh = show_xlam(rcat_r, show, dist=bool(i), ax=ax, colorby=colorby,
                        vmin=zmin, vmax=zmax, cmap=cmap,
                        marker='o', s=9, alpha=1.0, zorder=3, linewidth=0,)

    ax = axes[0, 0]
    ax.text(text[0], text[1], "H3 Giants",
            transform=ax.transAxes, bbox=bbox)

    # --- LM10 Mocks ---
    rprog = 0.66*0.85*rmax
    colorby, cname = rprog, r"$\hat{\rm R}_{\rm prog}$ (kpc)"
    vmin, vmax = 0.25, 2.5
    if config.by_vtan:
        colorby, cname = compute_vtan(lm10_r), r"V$_{\rm tan}$"
        vmin, vmax = -100, 500
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    #colorby, cname = lm10["tub"], r"t$_{\rm unbound}$"
    #vmin, vmax = 0, 5
    for i in range(2):
        ax = axes[1, i]
        show = unbound & (lm10["Pcol"] <= config.pcol_limit)
        #vmax = np.percentile(colorby[show], [84])[0]
        cbl = show_xlam(lm10_r, show, dist=bool(i), ax=ax, colorby=colorby,
                        vmin=vmin, vmax=vmax, cmap="magma_r",
                        marker='o', s=2, alpha=0.5, zorder=2, linewidth=0)

    cbl = ax.scatter([-10], [-10], c=[1], vmin=vmin, vmax=vmax, cmap="magma_r")
        # plot GCs
#        show = sgr_gcs
#        _ = show_xlam(gcat_r, show, dist=bool(i), ax=ax, colorby=None,
#                      color="cyan", marker='o', ms=5, alpha=1.0, zorder=3, linewidth=0,)

    ax = axes[1, 0]
    ax.text(text[0], text[1], "LM10\n(noiseless)",
            transform=ax.transAxes,bbox=bbox)

    # --- DL17 Mock ---
    colorby = dl17["id"]
    vmin, vmax = 0, 1
    cm = ListedColormap(["tomato", "black"])
    if config.by_vtan:
        colorby, cname =compute_vtan(dl17_r), r"V$_{\rm tan}$"
        vmin, vmax = -100, 500
        cm = "magma_r"

    for i in range(2):
        ax = axes[2, i]
        show = dl17["id"] >= 0
        cbd = show_xlam(dl17_r, show, dist=bool(i), ax=ax, colorby=colorby,
                        vmin=vmin, vmax=vmax, cmap=cm,
                        marker='o', s=2, alpha=1.0, zorder=2, linewidth=0)
    ax = axes[2, 0]
    ax.text(text[0], text[1], "DL17\n(noiseless)",
            transform=ax.transAxes, bbox=bbox)

    # --- prettify ---
    [ax.set_xlim(-5, 365) for ax in axes.flatten()]
    [ax.set_ylim(-330, 330) for ax in axes[:, 0]]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in axes[:, 0]]
    [ax.set_ylim(0, 90) for ax in axes[:, 1]]
    [ax.set_ylabel(r"$r_{\rm Gal}$ (kpc)") for ax in axes[:, 1]]
    [ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in axes[-1, :]]
    from matplotlib.lines import Line2D
    points = Line2D([], [], linestyle="", color="black",
                    marker="o", markersize=3)
    axes[0, 1].legend([points], ["[Fe/H] < {:.1f}".format(zcut)], loc="upper left", fontsize=10)

    # --- Colorbars ---
    cax1 = fig.add_subplot(gsc[1, -1])
    #pl.colorbar(cb, cax=cax, label=r"$t_{unbound}$ (Gyr)")
    pl.colorbar(cbl, cax=cax1, label=cname)
    cax2 = fig.add_subplot(gsc[0, -1])
    pl.colorbar(cbh, cax=cax2, label=r"[Fe/H]")
    cax3 = fig.add_subplot(gsc[2, -1])
    pl.colorbar(cbd, cax=cax3, label=r"", ticks=[0.25, 0.75])
    cax3.set_yticklabels(["Stars", "DM"])

    if config.savefig:
        fig.savefig("{}/x_lambda_mocks.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()