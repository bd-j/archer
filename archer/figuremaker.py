#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

from .config import rectify_config
from .catalogs import rectify, homogenize
from .frames import gc_frame_law10, gc_frame_dl17


class FigureMaker:

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
        self.lm10_rn = rectify(homogenize(self.lm10, "LM10", pcat=self.pcat,
                                          seds=self.lm10_seds, noisify_pms=config.noisify_pms,
                                          fractional_distance_error=config.fractional_distance_error),
                               gc_frame_law10)

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


if __name__ == "__main__":
    pass