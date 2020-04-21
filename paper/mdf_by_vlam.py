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
from archer.fitting import best_model, sample_posterior


if __name__ == "__main__":
    
    zmin, zmax = -3.0, 0.05
    zbins = np.arange(zmin, zmax, 0.1)

    try:
        parser.add_argument("--nsigma", type=float, default=2.)
    except:
        pass
    config = rectify_config(parser.parse_args())

    # rcat
    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
 
    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    n_tot = (good & sgr).sum()

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

    selections = {
        "$\Lambda<140$, Cold\n(Trailing Arm)": good & sgr & trail & cold_trail,
        "$\Lambda<140$, Diffuse": good & sgr & trail & ~cold_trail,
        "$\Lambda>200$, Cold\n(Leading Arm)": good & sgr & lead & cold_lead,
        "$\Lambda>200$, Diffuse": good & sgr & lead & ~cold_lead
    }

    # plot setup
    rcParams = plot_defaults(rcParams)
    figsize = (6.5, 6.5)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2,
                  left=0.08, right=0.94, wspace=0.2, hspace=0.3, top=0.95)

    axes = np.array([fig.add_subplot(g) for g in gs])
    axes = axes.reshape(2, 2).T

    for i, (k, sel) in enumerate(selections.items()):
        n_this = sel.sum()
        ax = axes.flat[i]
        ax.hist(rcat[sel]["FeH"], bins=zbins, color="darkgrey", alpha=0.8, histtype="stepfilled")
        ax.hist(rcat[sel]["FeH"], bins=zbins, color="darkgrey", histtype="step", linewidth=2)
        ax.text(0.06, 0.94, "{}\nN={}".format(k, sel.sum()),
                transform=ax.transAxes, verticalalignment="top")
        ax.hist(rcat[good & sgr]["FeH"], bins=zbins, weights=np.ones(n_tot) * n_this/n_tot,
                color="black", histtype="step", linestyle=":")

    [ax.set_xlabel("[Fe/H]") for ax in axes.flat]
    [ax.set_ylabel("N") for ax in axes.flat]
    
    if config.savefig:
        fig.savefig("{}/mdf_by_vlam.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()