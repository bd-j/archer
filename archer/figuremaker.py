#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

from .config import rectify_config
from .catalogs import rectify, homogenize
from .frames import gc_frame_law10, gc_frame_dl17
from .fitting import best_model
from .plotting import make_cuts


def show_lylz(cat, show, ax, colorby=None, **plot_kwargs):
    if colorby is not None:
        cb = ax.scatter(cat["lz"][show], cat["ly"][show], c=colorby[show],
                       **plot_kwargs)
        return ax, cb
    else:
        ax.plot(cat["lz"][show], cat["ly"][show],
                **plot_kwargs)
        return ax


class FigureMaker:

    ms = 2

    def __init__(self, args):
        self.config = rectify_config(args)
        self.setup(self.config)

    def plot_defaults(self, rcParams):
        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["STIXGeneral"]
        rcParams["font.size"] = 12
        rcParams["mathtext.fontset"] = "custom"
        rcParams["mathtext.rm"] = "serif"
        rcParams["mathtext.sf"] = "serif"
        rcParams['mathtext.it'] = 'serif:italic'
        return rcParams

    def setup(self, config):
        self.pcat = fits.getdata(config.pcat_file)
        self.get_rcat(config)
        self.get_lm10(config)
        self.get_dl17(config)

    def get_rcat(self, config):
        self.rcat = fits.getdata(self.config.rcat_file)
        self.rcat_r = rectify(homogenize(self.rcat, self.config.rcat_type,
                                         gaia_vers=self.config.gaia_vers),
                              self.config.gc_frame)

    def get_lm10(self, config):
        # lm10
        self.lm10 = fits.getdata(config.lm10_file)
        sedfile = os.path.join(os.path.dirname(config.lm10_file), "LM10_seds.fits")
        self.lm10_seds = fits.getdata(sedfile)
        self.lm10_r = rectify(homogenize(self.lm10, "LM10"), gc_frame_law10)

        # noisy lm10
        noisy = homogenize(self.lm10, "LM10", pcat=self.pcat, seds=self.lm10_seds,
                           noisify_pms=config.noisify_pms,
                           fractional_distance_error=config.fractional_distance_error)
        self.lm10_rn = rectify(noisy, gc_frame_law10)

    def get_dl17(self, config):
        # dl17
        self.dl17 = fits.getdata(config.dl17_file)
        self.dl17_r = rectify(homogenize(self.dl17, "DL17"), gc_frame_dl17)

    def get_gcs(self, config):
        self.gcat = fits.getdata(config.b19_file)
        self.gcat_r = rectify(homogenize(self.gcat, "B19"), config.gc_frame)

    def select(self, config, selector=None):
        good, sgr = selector(self.rcat, self.rcat_r, max_rank=config.max_rank,
                             dly=config.dly, flx=config.flx)
        self.good_sel = good
        self.sgr_sel = sgr
        return good, sgr

    def select_arms(self, nsigma=2, trailing_model="fits/h3_trailing_fit.h5",
                    leading_model="fits/h3_leading_fit.h5"):
        self.trailing_model = trailing_model
        self.leading_model = leading_model
        self.trail_sel = self.rcat_r["lambda"] < 175
        self.lead_sel =  self.rcat_r["lambda"] > 175

        tmu, tsig, _ = best_model(self.trailing_model, self.rcat_r["lambda"])
        self.cold_trail = np.abs(self.rcat_r["vgsr"] - tmu) < (nsigma * tsig)

        lmu, lsig, _ = best_model(self.leading_model, self.rcat_r["lambda"])
        self.cold_lead = np.abs(self.rcat_r["vgsr"] - lmu) < (nsigma * lsig)

        self.cold = (self.trail_sel & self.cold_trail) | (self.lead_sel & self.cold_lead)

    def break_axes(self, axes):
        # break axes with a little slash mark.
        [ax.spines['right'].set_visible(False) for ax in axes[:, 0]]
        [ax.spines['left'].set_visible(False) for ax in axes[:, 1]]
        [ax.yaxis.set_ticklabels([]) for ax in axes[:, 1]]
        [ax.yaxis.tick_left() for ax in axes[:, 0]]
        [ax.yaxis.tick_right() for ax in axes[:, 1]]
        _ = [make_cuts(ax, right=True, angle=2.0) for ax in axes[:, 0]]
        _ = [make_cuts(ax, right=False, angle=2.0) for ax in axes[:, 1]]

    def show_lylz(self, ax, show, label="", **kwargs):
        skwargs = dict(linestyle="", markersize=self.ms, marker="o", mew=0,
                       color='grey', alpha=0.5)
        skwargs.update(kwargs)
        ax = show_lylz(self.rcat_r, show, ax, label=label, **skwargs)


if __name__ == "__main__":
    pass