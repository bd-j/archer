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
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)

    # selections
    good = (rcat["FLAG"] == 0) & (rcat["SNR"] > 3) & (rcat["logg"] < 3.5)
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)
    
    trail = rcat_r["lambda"] < 175
    lead = rcat_r["lambda"] > 175
    arms = [trail, lead]

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
                  width_ratios=ncol * [10],
                  left=0.1, right=right, hspace=0.2, top=0.93)
                   #bottom=0.89, top=0.95)
    vlaxes = []
    cbars = []

    # --- plot H3 ----
    for iz, zrange in enumerate(zbins):
        for iarm, inarm in enumerate(arms):
            vlaxes.append(fig.add_subplot(gs[iz, iarm]))
            ax = vlaxes[-1]
            inz = (rcat["FeH"] < zrange[1]) & (rcat["FeH"] >= zrange[0])
            show = good & sgr & inz & inarm
            if colorby is not None:
                ax, cbh = show_vlam(rcat_r, show, ax=ax, colorby=colorby,
                                    vmin=zrange[0], vmax=zrange[1], cmap="magma",
                                    marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)
            else:
                ax, cbh = show_vlam(rcat_r, show, ax=ax, color="black", linestyle="", mew=0,
                                    marker='o', ms=2, alpha=0.9, zorder=2, linewidth=0)
            
            cbars.append(cbh)
    vlaxes = np.array(vlaxes).reshape(nrow, ncol)
    cbars = np.array(cbars).reshape(nrow, ncol)

    # prettify
    [ax.set_xlim(40, 140) for ax in vlaxes[:, 0]]
    [ax.set_xlim(200, 300) for ax in vlaxes[:, 1]]
    [ax.set_ylim(-330, 340) for ax in vlaxes.flat]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \cdot {\rm s}^{-1}$)") for ax in vlaxes[:, 0]]
    [ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in vlaxes[-1, :]]
    vlaxes[0, 0].set_title("Trailing")
    vlaxes[0, 1].set_title("Leading")

    # ---- Colorbars ----
    if colorby is not None:
        for iz, cb in enumerate(cbars[:, 1]):
            cax = fig.add_subplot(gsc[iz, -1])
            pl.colorbar(cb, cax=cax, label=r"[Fe/H]")
    else:
        for iz, ax in enumerate(vlaxes[:, 0]):
            ax.text(0.08, 0.08, "{} < [Fe/H] < {}".format(*zbins[iz]),
                    transform=ax.transAxes)

    if config.savefig:
        fig.savefig("{}/vgsr_lambda_feh.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
