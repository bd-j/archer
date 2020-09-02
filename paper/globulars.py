#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
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


if __name__ == "__main__":

    try:
        parser.add_argument("--feh_cut", type=float, default=-1.9)
    except:
        pass
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type
    zcut = config.feh_cut

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)
    lsgr = remnant_L(config.gc_frame)

    # GCs
    gcat = fits.getdata(config.b19_file)
    gcat_r = rectify(homogenize(gcat, "B19"), config.gc_frame)

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
    sgr_gcs, gc_feh = gc_select(gcat)
    #sgr_gcs = (gcat_r["ly"] < -2) & (gcat_r["ly"] > -7)
    unbound = lm10["tub"] > 0

    # plot setup
    rcParams = plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
    ms = 2
    text, bbox = [0.05, 0.1], dict(facecolor='white')
    nrow = 2
    figsize = (11, 5.0)  # 14, 8.5
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrow, 2, height_ratios=nrow * [10],
                  width_ratios=[3, 4],
                  hspace=0.15, wspace=0.2,
                  left=0.08, right=0.9,  bottom=0.12, top=0.95)
    gsc = GridSpec(1, 1, left=0.92, right=0.93, hspace=0.2,
                   bottom=0.12, top=0.95)
                   #bottom=0.89, top=0.95)
    axes = np.array([fig.add_subplot(gs[j, 1]) for j in range(nrow)])[None, :]


    # --- Plot ly-lz
    lax = fig.add_subplot(gs[:, 0])
    ax = show_lzly(rcat_r, good, lax, linestyle="",
                   marker="o", markersize=ms, mew=0, color='black', alpha=0.5)
    #ax.set_title("All H3 Giants")
    art = {"Sgr remnant (F18)": Line2D([], [], marker="*", ms=10, markerfacecolor="gold",
                                       markeredgecolor="k", linestyle=""),
           "H3 Giants": Line2D([], [], marker="o", ms=ms, mew=0, color="k", alpha=0.5, linestyle="")
           }

    # plot GCs
    ax = show_lzly(gcat_r, slice(None), lax, linestyle="",
                    marker="s", markersize=ms*2, markerfacecolor="tomato",
                    markeredgecolor='k', alpha=1.0)
    art["Globular Clusters (B19)"] = Line2D([], [], linestyle="", marker="s", markersize=ms*2,
                                        markerfacecolor="tomato", markeredgecolor='k')



    leg = list(art.keys())
    ax.legend([art[l] for l in leg], leg, fontsize=9)


    # --- plot x vs lambda ----
    colorby = rcat["feh"]
    zmin, zmax = -2, -0.1
    for i in range(2):
        ax = axes[0, i]
        show = good & sgr
        cbh = show_xlam(rcat_r, show, dist=bool(i), ax=ax, colorby=colorby,
                        vmin=zmin, vmax=zmax, cmap="magma",
                        marker='o', s=4, alpha=1.0, zorder=2, linewidth=0)
        # GCs
        show = sgr_gcs
        _ = show_xlam(gcat_r, show, dist=bool(i), ax=ax, colorby=gc_feh,
                        #color="cyan",
                        vmin=zmin, vmax=zmax, cmap="magma",
                        marker='s', s=36, alpha=1.0, zorder=3, linewidth=0.75, edgecolor="k")
        show = gcat["Name"] == "NGC 5466"
        _ = show_xlam(gcat_r, show, dist=bool(i), ax=ax, colorby=gc_feh,
                        #color="cyan",
                        vmin=zmin, vmax=zmax, cmap="magma",
                        marker='D', s=36, alpha=1.0, zorder=3, linewidth=0.75, edgecolor="k")

    # legend
    art = {"Sgr GCs": Line2D([], [], linestyle="", marker="s", markersize=6,
                                          markerfacecolor="none", mew=0.75, markeredgecolor='k'),
           "NGC5466": Line2D([], [], linestyle="", marker="D", markersize=6,
                                          markerfacecolor="none", mew=0.75, markeredgecolor='k')}
    leg = list(art.keys())
    axes[0,0].legend([art[l] for l in leg], leg, fontsize=9, loc = "lower left")

    #ax = axes[0, 0]
    #ax.text(text[0], text[1], "H3 Giants",
    #        transform=ax.transAxes, bbox=bbox)


    # plot 5824
    #show = gcat["name"] == "NGC 5824"
    #ax = show_lzly(gcat_r, show, lax, linestyle="",
    #                marker="s", markersize=ms*2, markerfacecolor="cyan",
    #                markeredgecolor='k', alpha=1.0)

    #for i, ax in enumerate(axes[0, :]):
    #    _ = show_xlam(gcat_r, show, dist=bool(i), ax=ax, colorby=gc_feh,
    #                    #color="cyan",
    #                    vmin=zmin, vmax=zmax, cmap="magma",
    #                    marker='o', s=36, alpha=1.0, zorder=3, linewidth=0.75, edgecolor="k")


    # --- plot selection line ---
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5 + config.dly, linestyle="--", color="royalblue", linewidth=2)
     for ax in [lax]]

    [ax.plot([lsgr[2]], [lsgr[1]], label="Sgr remnant", linestyle="",
             marker="*", markerfacecolor="gold", markersize=10, markeredgecolor="k",
             ) for ax in [lax]]

    # --- prettify ---
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit) for ax in [lax]]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit) for ax in [lax]]
    [ax.set_ylim(*span[1]) for ax in [lax]]
    [ax.set_xlim(*span[0]) for ax in [lax]]
    [ax.axvline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in [lax]]
    [ax.axhline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in [lax]]
    [ax.set_xlim(-5, 365) for ax in axes.flatten()]
    [ax.set_ylim(-330, 330) for ax in axes[:, 0]]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in axes[:, 0]]
    [ax.set_ylim(0, 93) for ax in axes[:, 1]]
    [ax.set_ylabel(r"$r_{\rm Gal}$ (kpc)") for ax in axes[:, 1]]
    [ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in axes[:, -1]]


    #points = Line2D([], [], linestyle="", color="black",
    #                marker="o", markersize=3)
    #axes[0, 0].legend([points], ["[Fe/H] < {}".format(zcut)], loc="upper right")

    # --- Colorbars ---
    cax1 = fig.add_subplot(gsc[0,0])
    pl.colorbar(cbh, cax=cax1, label=r"[Fe/H]")

    if config.savefig:
        fig.savefig("{}/sgr_globulars.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()