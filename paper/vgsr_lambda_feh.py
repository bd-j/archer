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
from archer.plotting import make_cuts
from archer.fitting import best_model, sample_posterior


def show_vlam(cat, show, ax=None, colorby=None, **plot_kwargs):
    if colorby is not None:
        cbh = ax.scatter(cat[show]["lambda"], cat[show]["vgsr"],
                         c=colorby[show], **plot_kwargs)
    else:
        cbh = ax.plot(cat[show]["lambda"], cat[show]["vgsr"],
                      **plot_kwargs)
    
    return ax, cbh


if __name__ == "__main__":

    zbins = [(-0.8, -0.1),
             (-1.9, -0.8),
             (-3, -1.9)]
    colorby = None
    #colorby = rcat["FeH"]

    try:
        parser.add_argument("--nsigma", type=float, default=2.)
    except:
        pass
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly)
   
    trail = rcat_r["lambda"] < 175
    lead = rcat_r["lambda"] > 175
    arms = [trail, lead]

    # trailing
    tsel = good & sgr & trail
    tmu, tsig, _ = best_model("fits/h3_trailing_fit.h5", rcat_r["lambda"])
    cold_trail = np.abs(rcat_r["vgsr"] - tmu) < (config.nsigma * tsig)
    
    # leading
    lsel = good & sgr & lead
    lmu, lsig, _ = best_model("fits/h3_leading_fit.h5", rcat_r["lambda"])
    cold_lead =  np.abs(rcat_r["vgsr"] - lmu) < (config.nsigma * lsig)

    cold = (lead & cold_lead) | (trail & cold_trail)

    # velocity fits
    tlam = np.sort(rcat_r[good & sgr & trail]["lambda"])
    llam = np.sort(rcat_r[good & sgr & lead]["lambda"])
    tmu, tsig, _ = best_model("fits/h3_trailing_fit.h5", tlam)
    lmu, lsig, _ = best_model("fits/h3_leading_fit.h5", llam)

    # plot setup
    rcParams = plot_defaults(rcParams)
    nrow, ncol = len(zbins), 2
    figsize = (9, 9)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    right = 0.95
    if colorby:
        right = 0.85
        gsc = GridSpec(nrow, 1, left=right, right=0.86, hspace=0.2)
    gs = GridSpec(nrow, ncol, height_ratios=nrow * [10],
                  width_ratios=ncol * [10], wspace=0.05,
                  left=0.09, right=right, hspace=0.2, top=0.95, bottom=0.09)
                   #bottom=0.89, top=0.95)
    vlaxes = []
    cbars = []

    # --- plot H3 ----
    for iz, zrange in enumerate(zbins):
        for iarm, inarm in enumerate(arms):
            vlaxes.append(fig.add_subplot(gs[iz, iarm]))
            ax = vlaxes[-1]
            inz = (rcat["FeH"] < zrange[1]) & (rcat["FeH"] >= zrange[0])
            show = good & sgr & inz & inarm & cold
            #if colorby is not None:
            #    ax, cbh = show_vlam(rcat_r, show, ax=ax, colorby=colorby,
            #                        vmin=zrange[0], vmax=zrange[1], cmap="magma",
            #                        marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
            ax, cbh = show_vlam(rcat_r, show, ax=ax, color="black", linestyle="", mew=0,
                                marker='o', ms=2, alpha=0.9, zorder=2, linewidth=0, label="Cold")
            show = good & sgr & inz & inarm & (~cold)
            ax, _ = show_vlam(rcat_r, show, ax=ax, color="tomato", linestyle="", mew=0.75,
                              marker='o', ms=2, alpha=1.0, zorder=2, linewidth=0.7, label="Diffuse",
                              fillstyle="none")
            cbars.append(cbh)

    vlaxes = np.array(vlaxes).reshape(nrow, ncol)
    cbars = np.array(cbars).reshape(nrow, ncol)


    # plot models
    if False:
        mkwargs = {"linestyle": "-", "color": "darkgrey", "linewidth": 1.0}
        [ax.plot(tlam, tmu + config.nsigma * tsig, **mkwargs) for ax in vlaxes[:, 0]]
        [ax.plot(tlam, tmu - config.nsigma * tsig, **mkwargs) for ax in vlaxes[:, 0]]
        [ax.plot(llam, lmu + config.nsigma * lsig, **mkwargs) for ax in vlaxes[:, 1]]
        [ax.plot(llam, lmu - config.nsigma * lsig, **mkwargs) for ax in vlaxes[:, 1]]

    # prettify
    vlaxes[0, 0].legend(loc="upper left", fontsize=10)
    
    [ax.set_xlim(40, 145) for ax in vlaxes[:, 0]]
    [ax.set_xlim(195, 300) for ax in vlaxes[:, 1]]
    [ax.set_ylim(-330, 340) for ax in vlaxes.flat]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in vlaxes[:, 0]]
    #[ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in vlaxes[-1, :]]
    #vlaxes[0, 0].set_title("Trailing")
    #vlaxes[0, 1].set_title("Leading")
    s1 = 0.48
    fig.text(s1, 0.03, r"$\Lambda_{\rm Sgr}$ (deg)")

    
    # break axes
    [ax.spines['right'].set_visible(False) for ax in vlaxes[:, 0]]
    [ax.spines['left'].set_visible(False) for ax in vlaxes[:, 1]]
    [ax.yaxis.set_ticklabels([]) for ax in vlaxes[:, 1]]
    [ax.yaxis.tick_left() for ax in vlaxes[:, 0]]
    #[ax.tick_params(labelright='off') for ax in vlaxes[:, 0]]
    [ax.yaxis.tick_right() for ax in vlaxes[:, 1]]
    _ = [make_cuts(ax, right=True, angle=2.0) for ax in vlaxes[:, 0]]
    _ = [make_cuts(ax, right=False, angle=2.0) for ax in vlaxes[:, 1]]

    # ---- Colorbars ----
    if colorby is not None:
        for iz, cb in enumerate(cbars[:, 1]):
            cax = fig.add_subplot(gsc[iz, -1])
            pl.colorbar(cb, cax=cax, label=r"[Fe/H]")
    else:
        for iz, ax in enumerate(vlaxes[:, 0]):
            ax.text(0.08, 0.08, "{:.1f} < [Fe/H] < {:.1f}".format(*zbins[iz]),
                    transform=ax.transAxes)

    if config.savefig:
        fig.savefig("{}/vgsr_lambda_feh.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
