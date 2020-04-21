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
from archer.catalogs import rectify, homogenize, pm_sigma
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

def remnant_L():
    from archer.frames import sgr_fritz18
    gc = sgr_fritz18.transform_to(config.gc_frame)
    xx = np.array([getattr(gc, a).to("kpc").value for a in "xyz"]).T
    p = np.array([getattr(gc, "v_{}".format(a)).to("km/s").value
                  for a in "xyz"]).T
    Lstar = np.cross(xx, p)
    return Lstar / 1e3


if __name__ == "__main__":

    try:
        parser.add_argument("--show_errors", action="store_true")
        parser.add_argument("--show_gcs", action="store_true")
        parser.add_argument("--ncol", type=int, default=2)
        parser.add_argument("--mag_cut", action="store_true")
    except:
        pass
    config = rectify_config(parser.parse_args())
    frac_err = config.fractional_distance_error
    ncol = config.ncol
    pcat = fits.getdata(config.pcat_file)

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
    lsgr = remnant_L()

    # GCs
    gcat = fits.getdata(config.b19_file)
    gcat_r = rectify(homogenize(gcat, "B19"), config.gc_frame)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    sedfile = os.path.join(os.path.dirname(config.lm10_file), "LM10_seds.fits")
    lm10_seds = fits.getdata(sedfile)
    lm10_r = rectify(homogenize(lm10, "LM10"), gc_frame_law10)

    # noisy lm10
    lm10_rn = rectify(homogenize(lm10, "LM10", pcat=pcat, 
                                 fractional_distance_error=frac_err), 
                      gc_frame_law10)
    if config.noisify_pms:
        pmunc = pm_sigma(lm10_seds, dist=lm10["dist"])
        lm10_rn["pmra"] += np.random.normal(size=len(lm10)) * pmunc[0]
        lm10_rn["pmdec"] += np.random.normal(size=len(lm10)) * pmunc[1]

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17"), gc_frame_dl17)

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = rcat_select(rcat, rcat_r)
    sgr_gcs, gc_feh = gc_select(gcat)
    unbound = lm10["tub"] > 0
    mag = lm10_seds["PS_r"] + 5 * np.log10(lm10_r["dist"])
    bright = (mag > 15) & (mag < 18.5)
    dl_remnant = ((dl17_r["ra"] < 315) & (dl17_r["ra"] > 285)  &
                  (dl17_r["dec"] < -25) & (dl17_r["dec"] > -32)) 

    # plot setup
    rcParams = plot_defaults(rcParams)
    span = [(-9.9, 11.5), (-14, 11)]
    ms = 2
    figsize = (4 * ncol + 2, 4.0)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, ncol, width_ratios=ncol * [10],
                  left=0.06, right=0.98, wspace=0.25, top=0.93, bottom=0.13)
    #gsc = GridSpec(1, 2, left=0.1, right=0.95, wspace=0.3,
    #               bottom=0.89, top=0.95)
    laxes = []

    # --- plot H3 ---
    laxes.append(fig.add_subplot(gs[0, 0]))
    ax = show_lzly(rcat_r, good, laxes[0], linestyle="",
                   marker="o", markersize=ms, mew=0, color='black', alpha=0.5)
    if config.show_errors:
        show_ellipses(rcat[good & sgr], rcat_r[good & sgr], ax=ax, 
                      covdir=config.covar_dir, alpha=0.3)

    ax.set_title("All H3 Giants")
    art = {"Sgr remnant (F18)": Line2D([], [], marker="*", ms=8, markerfacecolor="royalblue",
                                       markeredgecolor="k", linestyle=""),
           }

    # plot GCs
    if config.show_gcs:
        ax = show_lzly(gcat_r, slice(None), laxes[0], linestyle="",
                       marker="s", markersize=ms*2, markerfacecolor="tomato",
                       markeredgecolor='k', alpha=1.0)
        art["Globular Clusters"] = Line2D([], [], linestyle="", marker="s", markersize=ms*2,
                                          markerfacecolor="tomato", markeredgecolor='k')
    
    leg = list(art.keys())
    ax.legend([art[l] for l in leg], leg, fontsize=10)
    
    # --- plot LM10 ---
    laxes.append(fig.add_subplot(gs[0, 1], sharey=laxes[0], sharex=laxes[0]))
    show = unbound & (lm10_rn["in_h3"] == 1)
    if config.mag_cut:
        show = show & bright
    ax = show_lzly(lm10_rn, show, laxes[-1], linestyle="",
                   marker="o", markersize=ms, mew=0, color='grey',
                   alpha=0.5, zorder=0)
    show = unbound
    _ = twodhist(lm10_r["lz"][show], lm10_r["ly"][show], ax=laxes[-1],
                  span=span, fill_contours=False, color="black",
                  contour_kwargs={"linewidths": 0.75})
    ax.set_title("LM10")
    art = {"Unbound particles": Line2D([], [], color="k", linewidth=0.75),
           "Within H3 window": Line2D([], [], marker="o", markersize=ms, linewidth=0, color="grey")}
    leg = list(art.keys())
    ax.legend([art[l] for l in leg], leg, fontsize=10)

    # --- plot DL17 ---
    if ncol > 2:
        colorby = dl17["id"]
        vmin, vmax = 0, 1
        cm = ListedColormap(["tomato", "black"])
        rand = np.random.choice(len(dl17), size=(good & sgr).sum(), replace=False)

        ax = fig.add_subplot(gs[0, 2], sharey=laxes[0], sharex=laxes[0])
        laxes.append(ax)
        #ax, cb = show_lzly(dl17_r, rand, laxes[-1], colorby=colorby,
        #                   vmin=vmin, vmax=vmax, cmap=cm,
        #                   marker="o", s=ms**2, linewidth=0, alpha=0.8)
        
        stars = (dl17["id"] < 1) & ~dl_remnant
        dark = (dl17["id"] == 1) & ~dl_remnant
        _ = twodhist(dl17_r["lz"][stars], dl17_r["ly"][stars], ax=laxes[-1],
                     span=span, fill_contours=True, color="tomato",
                     contour_kwargs={"linewidths": 1.0})
        _ = twodhist(dl17_r["lz"][dark], dl17_r["ly"][dark], ax=laxes[-1],
                     span=span, fill_contours=False, color="k",
                     contour_kwargs={"linewidths": 0.75})
        ax.plot([np.nanmedian(dl17_r["lz"][dl_remnant])], [np.nanmedian(dl17_r["ly"][dl_remnant])],
                marker="*", markersize=7, linewidth=0, color="k", label="DL17 remnant")

        ax.set_title("DL17")
        art = {"Unbound stars": Line2D([], [], color="tomato"),
               "Dark Matter": Line2D([], [], color="k", linewidth=0.75),
               "Remnant": Line2D([], [], marker="*", markersize=7, linewidth=0, color="k")}
        leg = list(art.keys())
        ax.legend([art[l] for l in leg], leg, fontsize=10)
    
    # --- plot selection line ---
    zz =  np.linspace(-9, 10, 100)
    [ax.plot(zz, -0.3 * zz - 2.5, linestyle="--", color="royalblue", linewidth=2) for ax in laxes]

    [ax.plot([lsgr[2]], [lsgr[1]], label="Sgr remnant", linestyle="",
             marker="*", markerfacecolor="royalblue", markersize=8, markeredgecolor="k",
             ) for ax in laxes]

    # --- prettify ---
    lunit = r" ($10^3 \,\, {\rm kpc} \,\, {\rm km} \,\, {\rm s}^{-1}$)"
    [ax.set_ylabel(r"L$_{\rm y}$" + lunit) for ax in laxes]
    [ax.set_xlabel(r"L$_{\rm z}$" + lunit) for ax in laxes]
    [ax.set_ylim(*span[1]) for ax in laxes]
    [ax.set_xlim(*span[0]) for ax in laxes]
    [ax.axvline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in laxes]
    [ax.axhline(0, linestyle="-", color="k", linewidth=0.75, alpha=0.8) for ax in laxes]


    if config.savefig:
        fig.savefig("{}/selection_lylz.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()