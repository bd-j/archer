#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.frames import gc_frame_law10, gc_frame_dl17
from archer.plotting import make_cuts
from archer.plummer import convert_estar_rmax


def show_allx(cat_r, selection, colorby=None, nshow=None,
              splitlambda=175, icat=0,
              figure=None, gridspec=None, **plot_kwargs):

    pkwargs = dict(vmin=-2.5, vmax=-0.5, cmap="rainbow",
                   marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
    pkwargs.update(plot_kwargs)

    sel = selection
    if nshow is None:
        nshow = sel.sum()
    rand = np.random.choice(sel.sum(), size=nshow, replace=False)

    rgal = np.sqrt(cat_r["x_gal"]**2 + cat_r["y_gal"]**2 + cat_r["z_gal"]**2)
    ycols = cat_r["vgsr"], rgal
    arms = (cat_r["lambda"] < splitlambda), (cat_r["lambda"] > splitlambda)

    axes, cbars = [], []
    for iy, ycol in enumerate(ycols):
        gs = gridspec[iy]

        for iarm, arm in enumerate(arms):

            #j = iy * 2 + iarm
            ax = figure.add_subplot(gs[icat, iarm])

            xx = cat_r["lambda"][sel][rand]
            yy = ycol[sel][rand]
            zz = colorby[sel][rand]
            inarm = arm[sel][rand]

            cb = ax.scatter(xx[inarm], yy[inarm], c=zz[inarm], **pkwargs)

            axes += [ax]
            cbars += [cb]

    return axes, cbars


if __name__ == "__main__":

    np.random.seed(101)
    try:
        parser.add_argument("--show_gcs", action="store_true")
        parser.add_argument("--extra_sigma", type=float, default=40)
        parser.add_argument("--outer_radius", type=float, default=1.5)
    except:
        pass

    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type
    frac_err = config.fractional_distance_error

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)
    pcat = fits.getdata(config.pcat_file)

    # lm10
    lm10 = fits.getdata(config.lm10_file)
    sedfile = os.path.join(os.path.dirname(config.lm10_file), "LM10_seds.fits")
    lm10_seds = fits.getdata(sedfile)
    rmax, energy = convert_estar_rmax(lm10["estar"])
    lm10_noiseless = rectify(homogenize(lm10, "LM10"), gc_frame_law10)

    # noisy lm10
    lm10_r = rectify(homogenize(lm10, "LM10", pcat=pcat,
                                 seds=lm10_seds, noisify_pms=config.noisify_pms,
                                 fractional_distance_error=frac_err), 
                      gc_frame_law10)

    # add dispersion to outer LM10 particles
    outer = (0.66*0.85*rmax) > config.outer_radius
    dv = outer * config.extra_sigma * np.random.normal(size=len(lm10))
    lm10_r["vgsr"] += dv

    # dl17
    dl17 = fits.getdata(config.dl17_file)
    dl17_r = rectify(homogenize(dl17, "DL17", pcat=pcat,
                                fractional_distance_error=frac_err),
                     gc_frame_dl17)

    # selections
    from make_selection import rcat_select, gc_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly, flx=config.flx)
    unbound = lm10["tub"] > 0
    mag = lm10_seds["PS_r"] + 5 * np.log10(lm10_noiseless["dist"])
    bright = (mag > 15) & (mag < 18.5)

    # plot setup
    rcParams = plot_defaults(rcParams)
    text = [0.1, 0.1]
    bbox = dict(facecolor='white')
    nrow = 3
    figsize = (11, 6.6)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gsv = GridSpec(nrow, 2, height_ratios=nrow * [10],
                   hspace=0.2, wspace=0.08,
                   left=0.08, right=0.46, top=0.93, bottom=0.08)
    gsd = GridSpec(nrow, 2, height_ratios=nrow * [10],
                   hspace=0.2, wspace=0.08,
                   left=0.52, right=0.90, top=0.93, bottom=0.08)
    gsc = GridSpec(nrow, 1, hspace=0.2,
                   left=0.92, right=0.93, top=0.93, bottom=0.08)
    vlaxes, vcb = [], []

    # --- plot H3 ----
    axes, cbs = show_allx(rcat_r, good & sgr, colorby=rcat["FeH"],
                          icat=0, nshow=None, figure=fig, gridspec=(gsv, gsd),
                          vmin=-2.0, vmax=-0.1, cmap="magma",
                          marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
    nshow = (good & sgr).sum()
    vlaxes.append(axes)
    vcb.append(cbs[0])

    # --- LM10 Mocks ---
    colorby, cname = 0.66*0.85*rmax, r"$\hat{\rm R}_{\rm prog}$ (kpc)" #r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)"
    vmin, vmax = 0.25, 2.5
    #colorby, cname = lm10["Estar"], r"E$_\ast$"
    #vmin, vmax = 0, 1
    sel = unbound & (lm10_r["in_h3"] == 1)
    if config.mag_cut:
        sel = sel & bright
    axes, cbs = show_allx(lm10_r, sel, colorby=colorby,
                          icat=1, nshow=nshow, figure=fig, gridspec=(gsv, gsd),
                          vmin=vmin, vmax=vmax, cmap="magma_r",
                          marker='o', linewidth=0, alpha=1.0, s=4)
    vlaxes.append(axes)
    vcb.append(cbs[0])

    # --- DL17 Mock ---
    cm = ListedColormap(["tomato", "black"])
    sel = (dl17["id"] >= 0) & (dl17_r["in_h3"] == 1)
    axes, cbs = show_allx(dl17_r, sel, colorby=dl17["id"],
                          icat=2, nshow=nshow, figure=fig, gridspec=(gsv, gsd),
                          vmin=0, vmax=1, cmap=cm,
                          marker='o', linewidth=0, alpha=1.0, s=4)
    vlaxes.append(axes)
    vcb.append(cbs[0])

    vlaxes = np.array(vlaxes)

    # prettify
    [ax.set_xlim(40, 145) for ax in vlaxes[:, 0::2].flat]
    [ax.set_xlim(195, 300) for ax in vlaxes[:, 1::2].flat]
    [ax.set_ylim(-330, 330) for ax in vlaxes[:, :2].flat]
    [ax.set_ylim(0, 90) for ax in vlaxes[:, 2:].flat]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)")
     for ax in vlaxes[:, 0].flat]
    [ax.set_ylabel(r"$R_{\rm GC}$ (kpc)") for ax in vlaxes[:, 2].flat]
    #[ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in vlaxes[-1,:]]

    # break axes
    [ax.spines['right'].set_visible(False) for ax in vlaxes[:, 0::2].flat]
    [ax.spines['left'].set_visible(False) for ax in vlaxes[:, 1::2].flat]
    [ax.yaxis.set_ticklabels([]) for ax in vlaxes[:, 1::2].flat]
    [ax.yaxis.tick_left() for ax in vlaxes[:, 0::2].flat]
    #[ax.tick_params(labelright='off') for ax in vlaxes[:, 0]]
    #[ax.yaxis.set_label_position("right") for ax in vlaxes[:, -1]]
    [ax.yaxis.tick_right() for ax in vlaxes[:, 1::2].flat]
    _ = [make_cuts(ax, right=True, angle=2.0) for ax in vlaxes[:, 0::2].flat]
    _ = [make_cuts(ax, right=False, angle=2.0) for ax in vlaxes[:, 1::2].flat]

    # Labels
    [ax.text(text[0], text[1], "H3", transform=ax.transAxes, bbox=bbox)
     for ax in vlaxes[0, 0:1]]
    [ax.text(text[0], text[1], "LM10", transform=ax.transAxes, bbox=bbox)
     for ax in vlaxes[1, 0:1]]
    [ax.text(text[0], text[1], "DL17", transform=ax.transAxes, bbox=bbox)
     for ax in vlaxes[2, 0:1]]

    s1 = 0.25
    fig.text(s1, 0.03, r"$\Lambda_{\rm Sgr}$ (deg)")
    fig.text(s1 + 0.4 + 0.04, 0.03, r"$\Lambda_{\rm Sgr}$ (deg)")

    # ---- Colorbars ----
    cax1 = fig.add_subplot(gsc[1, -1])
    #pl.colorbar(cb, cax=cax, label=r"$t_{unbound}$ (Gyr)")
    cb1 = pl.colorbar(vcb[1], cax=cax1,)
    cb1.ax.set_ylabel(cname, rotation=90, clip_on=False)
    cax2 = fig.add_subplot(gsc[0, -1])
    pl.colorbar(vcb[0], cax=cax2, label=r"[Fe/H]")
    cax3 = fig.add_subplot(gsc[2, -1])
    pl.colorbar(vcb[2], cax=cax3, label=r"", ticks=[0.25, 0.75])
    cax3.set_yticklabels(["Stars", "DM"])

    if config.savefig:
        fig.savefig("{}/extra_sigma.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()
