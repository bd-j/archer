#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib import colors
from matplotlib.colors import ListedColormap

from astropy.io import fits

from archer.config import parser
from archer.fitting import best_model, sample_posterior

from archer.figuremaker import FigureMaker


def show_vlam(cat, show, ax=None, colorby=None, **plot_kwargs):
    if colorby is not None:
        cbh = ax.scatter(cat[show]["lambda"], cat[show]["vgsr"],
                         c=colorby[show], **plot_kwargs)
    else:
        cbh, = ax.plot(cat[show]["lambda"], cat[show]["vgsr"],
                      **plot_kwargs)

    return ax, cbh


class Plotter(FigureMaker):

    def make_axes(self, nrow=3, ncol=2, figsize=(9, 9), colorby=False):
        self.fig = pl.figure(figsize=figsize)
        from matplotlib.gridspec import GridSpec
        right = 0.95
        if colorby:
            right = 0.85
            self.gsc = GridSpec(nrow, 1, left=right, right=0.86, hspace=0.2)
        self.gs = GridSpec(nrow, ncol, height_ratios=nrow * [10],
                           width_ratios=ncol * [10], wspace=0.05,
                           left=0.09, right=right, hspace=0.2, top=0.95, bottom=0.09)
                           #bottom=0.89, top=0.95)
        vlaxes = [self.fig.add_subplot(self.gs[i, j]) for i in range(nrow) for j in range(ncol)]
        self.axes = np.array(vlaxes).reshape(nrow, ncol)

        self.break_axes(self.axes)

    def read_velocity_fits(self):
        pass

    def plot_zbin(self, axes, zrange, diffcolor="crimson"):

        face = colors.to_rgb(diffcolor)
        face = tuple(list(face) + [0.25])
        with np.errstate(invalid="ignore"):
            inz = (self.rcat["FeH"] < zrange[1]) & (self.rcat["FeH"] >= zrange[0])

        cbars = []
        # --- plot H3 ----
        aname = ["trail", "lead"]
        arms = [self.trail_sel, self.lead_sel]
        for iarm, inarm in enumerate(arms):
            ax = axes[iarm]
            #if colorby is not None:
            #    ax, cbh = show_vlam(self.rcat_r, show, ax=ax, colorby=colorby,
            #                        vmin=zrange[0], vmax=zrange[1], cmap="magma",
            #                        marker='o', s=4, alpha=0.8, zorder=2, linewidth=0)

            # --- cold ---
            show = self.good_sel & self.sgr_sel & inz & inarm & self.cold
            print("{} cold: {:.0f}".format(aname[iarm], show.sum()))
            ax, cbh = show_vlam(self.rcat_r, show, ax=ax, color="black", linestyle="", mew=0,
                                marker='o', ms=2, alpha=1.0, zorder=2, linewidth=0, label="Cold")

            # --- diffuse ---
            show = self.good_sel & self.sgr_sel & inz & inarm & (~self.cold)
            print("{} diffuse: {:.0f}".format(aname[iarm], show.sum()))
            ax, cb = show_vlam(self.rcat_r, show, ax=ax, color="black", linestyle="", mew=0,
                               markeredgecolor=diffcolor, markerfacecolor=face,
                               #fillstyle="none",
                               marker='o', ms=2, zorder=2, alpha=1.0, linewidth=0, label="Diffuse",)
            cb.set_markerfacecolor(face)
            cbars.append(cbh)

        return cbars

    def show_velmodel(self, axes, trail=True, nsigma=2, **lkwargs):
        mkwargs = dict(linestyle= "-", color="darkgrey", linewidth=1.0)
        mkwargs.update(lkwargs)

        if trail:
            arm = self.trail_sel
            model = self.trailing_model
        else:
            arm = self.lead_sel
            model = self.leading_model

        lam = np.sort(self.rcat_r[self.good_sel & self.sgr_sel & arm]["lambda"])
        mu, sig, _ =  best_model(model, lam)
        [ax.plot(lam, mu + nsigma * sig, **mkwargs) for ax in axes]

    def select_arms(self, nsigma=2):
        self.trailing_model = "fits/h3_trailing_fit.h5"
        self.leading_model = "fits/h3_leading_fit.h5"
        self.trail_sel = self.rcat_r["lambda"] < 175
        self.lead_sel =  self.rcat_r["lambda"] > 175

        tmu, tsig, _ = best_model(self.trailing_model, self.rcat_r["lambda"])
        self.cold_trail = np.abs(self.rcat_r["vgsr"] - tmu) < (nsigma * tsig)

        lmu, lsig, _ = best_model(self.leading_model, self.rcat_r["lambda"])
        self.cold_lead = np.abs(self.rcat_r["vgsr"] - lmu) < (nsigma * lsig)

        self.cold = (self.trail_sel & self.cold_trail) | (self.lead_sel & self.cold_lead)


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

    args = parser.parse_args()
    plotter = Plotter(args)
    config = plotter.config

    # selections
    from make_selection import rcat_select
    good, sgr = plotter.select(config, selector=rcat_select)
    plotter.select_arms(nsigma=args.nsigma)

    # plot setup
    nrow, ncol = len(zbins), 2
    plotter.plot_defaults(rcParams)
    plotter.make_axes(nrow, ncol, figsize=(9, 9))

    # plot data
    cbars = []
    for iz, zrange in enumerate(zbins):
        cb = plotter.plot_zbin(plotter.axes[iz, :], zrange)
        cbars.append(cb)
    cbars = np.array(cbars)

    # plot models
    if False:
        plotter.show_velmodel(plotter.axes[:, 0], trail=True, nsigma=config.nsigma)
        plotter.show_velmodel(plotter.axes[:, 1], trail=False, nsigma=config.nsigma)

    # prettify
    plotter.axes[0, 0].legend(loc="upper left", fontsize=10)
    [ax.set_xlim(40, 145) for ax in plotter.axes[:, 0]]
    [ax.set_xlim(195, 300) for ax in plotter.axes[:, 1]]
    [ax.set_ylim(-330, 340) for ax in plotter.axes.flat]
    [ax.set_ylabel(r"V$_{\rm GSR}$ (${\rm km} \,\, {\rm s}^{-1}$)") for ax in plotter.axes[:, 0]]

    s1 = 0.48
    plotter.fig.text(s1, 0.03, r"$\Lambda_{\rm Sgr}$ (deg)")


    # ---- Colorbars ----
    if colorby is not None:
        for iz, cb in enumerate(cbars[:, 1]):
            cax = plotter.fig.add_subplot(plotter.gsc[iz, -1])
            pl.colorbar(cb, cax=cax, label=r"[Fe/H]")
    else:
        for iz, ax in enumerate(plotter.axes[:, 0]):
            ax.text(0.08, 0.08, "{:.1f} < [Fe/H] < {:.1f}".format(*zbins[iz]),
                    transform=ax.transAxes)

    if config.savefig:
        plotter.fig.savefig("{}/vgsr_lambda_feh.{}".format(config.figure_dir, config.figure_extension),
                    dpi=config.figure_dpi)
