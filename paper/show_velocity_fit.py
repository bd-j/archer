#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from astropy.io import fits

from archer.config import parser, rectify_config, plot_defaults
from archer.catalogs import rectify, homogenize
from archer.fitting import best_model, sample_posterior
from archer.plotting import make_cuts


def show_vfit(lam, mu, sig, nsig=1, ax=None, color="k", 
              plot_kwargs={}, fill_kwargs={}):
    o = np.argsort(lam)
    ax.plot(lam[o], mu[o], color=color, **plot_kwargs)
    if nsig > 0:
        ax.fill_between(lam[o], mu[o] - nsig * sig[o], mu[o] + nsig * sig[o],
                        color=color, **fill_kwargs)
    return ax


def show_sfit(lam, sig, ax=None, samples=None, color="k",
              plot_kwargs={}, fill_kwargs={}):
    o = np.argsort(lam)
    ax.plot(lam[o], sig[o], color=color, **plot_kwargs)
    if samples is not None:
        sigsig = samples.std(axis=0)
        ax.fill_between(lam[o], sig[o] - nsig * sigsig[o], sig[o] + nsig * sigsig[o],
                        color=color, **fill_kwargs)
    return ax


# --- Literature traces ---
g17trail = [(240, 250, 260, 270),
            (15.7, 13.5, 12.5, 12.6), (1.5, 1.1, 1.1, 1.5),
            (14.0, 8.5, 7.1, 6.4), (2.6, 1.3, 1.4, 3.0),
            (-142.5, -124.2, -107.4, -79.0),
            (-143.5, -114.7, -98.6, -75.7)]
g17lead = [(105, 115, 125, 135),
           (31.4, 21.6, 15.0, 16.9), (6.0, 4.0, 4.0, 3.2),
           (19.7, 12.0, 6.0, 10.5), (5.0, 2.0, 1.5, 2.0),
           (-87.6, -107.2, -110.7, -123.8),
           (-77.5, -90.2, -107.1, -116.7)]

cols = ["lam", "vsig1", "vsig1_err", "vsig2", "vsig2_err", "vel1", "vel2"]
dt = np.dtype([(n, np.float) for n in cols])
gibt = np.zeros(len(g17trail[0]), dtype=dt)
for d, c in zip(g17trail, cols):
    gibt[c] = d

gibl = np.zeros(len(g17lead[0]), dtype=dt)
for d, c in zip(g17lead, cols):
    gibl[c] = d


belokurov14 = [(217.5, 227.5, 232.5, 237.5, 242.5, 247.5, 252.5, 257.5, 262.5,
                267.5, 272.5, 277.5, 285.0, 292.5),
               (-127.2, -141.1, -150.8, -141.9, -135.1, -129.5, -120.0, -108.8,
                -98.6, -87.2, -71.8, -58.8, -35.4, -7.8)]

if __name__ == "__main__":

    nsig = 2
    config = rectify_config(parser.parse_args())
    rtype = config.rcat_type

    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, rtype), config.gc_frame)
    
    # selection
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r, dly=config.dly)
    n_tot = (good & sgr).sum()

    trail = rcat_r["lambda"] < 175
    lead = rcat_r["lambda"] > 175
    arms = [trail, lead]

    # trailing
    tsel = good & sgr & trail
    gt = rcat_r["lambda"] > 65
    gl = rcat_r["lambda"] > 220
    tmu, tsig, _ = best_model("fits/h3_trailing_fit.h5", rcat_r["lambda"])
    tmus, tsigs, tpars = sample_posterior("fits/h3_trailing_fit.h5", rcat_r["lambda"])
    mock_tmu, mock_tsig, _ = best_model("fits/lm10_trailing_fit.h5", rcat_r["lambda"])

    # leading
    lsel = good & sgr & lead
    lmu, lsig, _ = best_model("fits/h3_leading_fit.h5", rcat_r["lambda"])
    lmus, lsigs, lpars = sample_posterior("fits/h3_leading_fit.h5", rcat_r["lambda"])
    mock_lmu, mock_lsig, _ = best_model("fits/lm10_leading_fit.h5", rcat_r["lambda"])

    order = np.argsort(rcat_r["lambda"])

    # plot setup
    ncol = 2
    rcParams = plot_defaults(rcParams)
    figsize = (5., 6.2)
    fig = pl.figure(figsize=figsize)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(ncol, 2, height_ratios=ncol * [10],
                  wspace=0.05, hspace=0.15, bottom=0.09,
                  left=0.14, right=0.92, top=0.93)

    #gsc = GridSpec(ncol, 1, left=0.85, right=0.86, hspace=0.2)
    vtax = fig.add_subplot(gs[0, 0])
    vlax = fig.add_subplot(gs[0, 1])
    stax = fig.add_subplot(gs[1, 0])
    slax = fig.add_subplot(gs[1, 1])
    
    axes = np.array([[vtax, vlax], [stax, slax]])
    
    lmcolor = "tomato"
    fcolor = "black"
    cmap = "magma"
    fillkw = dict(alpha=0.2, linewidth=0)

    # plot fits
    show_vfit(rcat_r["lambda"][tsel & gt], tmu[tsel & gt], tsig[tsel & gt],
              color=fcolor, ax=vtax, fill_kwargs=fillkw)
    #show_vfit(rcat_r["lambda"][tsel & gt], mock_tmu[tsel & gt], mock_tsig[tsel & gt],
    #          nsig=0, color=lmcolor, ax=vtax, plot_kwargs=dict(linestyle="--"))
    show_vfit(rcat_r["lambda"][lsel], lmu[lsel], lsig[lsel],
              color=fcolor, ax=vlax, fill_kwargs=fillkw)
    #show_vfit(rcat_r["lambda"][lsel& gl], mock_lmu[lsel & gl], mock_lsig[lsel & gl],
    #          nsig=0, color=lmcolor, ax=vlax, plot_kwargs=dict(linestyle="--"))
    
    # plot sigmas
    show_sfit(rcat_r["lambda"][tsel & gt], tsig[tsel & gt], samples=tsigs[:, tsel & gt],
              ax=stax, color=fcolor, fill_kwargs=fillkw)
    show_sfit(rcat_r["lambda"][tsel & gt], mock_tsig[tsel & gt],
              ax=stax, color=lmcolor, plot_kwargs=dict(linestyle="--", label="fit to LM10"))    
    show_sfit(rcat_r["lambda"][lsel], lsig[lsel], samples=lsigs[:, lsel],
              ax=slax, color=fcolor, fill_kwargs=fillkw)
    show_sfit(rcat_r["lambda"][lsel & gl], mock_lsig[lsel & gl],
              ax=slax, color=lmcolor, plot_kwargs=dict(linestyle="--"))    


    # plot data points
    #sel = good & sgr
    cbt = vtax.plot(rcat_r[tsel]["lambda"], rcat_r[tsel]["vgsr"],
                    #c=rcat[tsel]["FeH"], vmin=-2., vmax=-0.1, cmap=cmap,
                    marker="o", color="k", markersize=1, linestyle="")
    cbl = vlax.plot(rcat_r[lsel]["lambda"], rcat_r[lsel]["vgsr"],
                    #c=rcat[lsel]["FeH"], vmin=-2., vmax=-0.1, cmap=cmap,
                    marker="o", color="k", markersize=1, linestyle="")

    # Monaco and Majewski
    stax.fill_between(np.array([30, 90]), 8.3 - 0.9, 8.3 + 0.9, alpha=0.5,
                      label="Monaco07", color="royalblue")
    stax.plot(np.array([30, 90]), np.array([11.7, 11.7]), linestyle="--",
              label="Majewski04", color="royalblue")



    # Plot the gibbons and belokurov trends
    kw = dict(marker="o", alpha=0.6, markersize=3)
    mp = "darkslateblue"
    mr = "maroon"
    #vlax.plot(360 - gibl["lam"], gibl["vel1"], label="G17 (Metal Poor)",
    #          alpha=1, color="lightblue")
    #vlax.plot(360 - gibl["lam"], gibl["vel2"], label="G17 (Metal Rich)",
    #            alpha=1, color="maroon")
    slax.errorbar(360 - gibl["lam"], gibl["vsig1"], yerr=gibl["vsig1_err"],
                  label="G17 (Metal Poor)",  color=mp, **kw)
    slax.errorbar(360 - gibl["lam"], gibl["vsig2"], yerr=gibl["vsig2_err"],
                  label="G17 (Metal Rich)", color=mr, **kw)

    #vtax.plot(360 - gibt["lam"], gibt["vel1"], label="G17 (Metal Poor)",
    #          alpha=1, color="lightblue")
    #vtax.plot(360 - gibt["lam"], gibt["vel2"], label="G17 (Metal Rich)",
    #          alpha=1, color="maroon")

    stax.errorbar(360 - gibt["lam"], gibt["vsig1"], yerr=gibt["vsig1_err"],
                  label="G17 (Metal Poor)", color=mp, **kw)
    stax.errorbar(360 - gibt["lam"], gibt["vsig2"], yerr=gibt["vsig2_err"],
                  label="G17 (Metal Rich)", color=mr, **kw)


    # Prettify
    [ax.set_xlim(40, 145) for ax in axes[:, 0]]
    [ax.set_xlim(195, 300) for ax in axes[:, 1]]
    [ax.set_ylim(-330, 340) for ax in axes[0, :]]
    [ax.set_ylim(0, 45) for ax in axes[1, :]]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in axes[0, 0:1]]
    [ax.set_ylabel(r"$\sigma_{\rm V}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in axes[1, 0:1]]
    #[ax.set_xlabel(r"$\Lambda_{\rm Sgr}$ (deg)") for ax in axes[-1, :]]
    axes[0, 0].set_title("Trailing")
    axes[0, 1].set_title("Leading")
    s1 = 0.48
    fig.text(s1, 0.022, r"$\Lambda_{\rm Sgr}$ (deg)")


    # break axes
    [ax.spines['right'].set_visible(False) for ax in axes[:, 0]]
    [ax.spines['left'].set_visible(False) for ax in axes[:, 1]]
    [ax.yaxis.set_ticklabels([]) for ax in axes[:, 1]]
    [ax.yaxis.tick_left() for ax in axes[:, 0]]
    #[ax.tick_params(labelright='off') for ax in vlaxes[:, 0]]
    [ax.yaxis.tick_right() for ax in axes[:, 1]]
    _ = [make_cuts(ax, right=True, angle=2.0) for ax in axes[:, 0]]
    _ = [make_cuts(ax, right=False, angle=2.0) for ax in axes[:, 1]]

    stax.legend(fontsize=9, loc="upper left")
    
    #cax = fig.add_subplot(gsc[0, 0])
    #pl.colorbar(cbh, cax=cax, label=r"[Fe/H]")#, orientation="horizontal")

    if config.savefig:
        fig.savefig("{}/vel_fit.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
    else:
        pl.show()