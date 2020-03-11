#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from argparse import Namespace
import astropy.coordinates

epath = os.path.expandvars
pjoin = os.path.join

config = Namespace()

# --- Figures ---
config.figure_extension = "png"
config.dpi = 450
config.savefigs = False


# --- catalog versions ---
config.add_noise = False
config.segue_cat = False
config.data = epath("$HOME/Projects/archer/data/")

config.lm10_file = pjoin(config.data, "mocks", "LM10", "SgrTriax.fits")
config.dl17_file = pjoin(config.data, "mocks", "DL17", "DL17_all_orig.fits")
config.r18_file = pjoin(config.data, "mocks", "R18", "R18_noiseless_v5.fits")

config.segue_file = pjoin(config.data, "catalogs", "ksegue_gaia_v5.fits")
config.rcat_vers = "2_0"
config.rcat_file = pjoin(config.data, "catalogs", "rcat_V{}_MSG.fits".format(config.rcat_vers.replace("_", ".")))

# --- kinematics ---
config.gc_frame = astropy.coordinates.Galactocentric()


def plot_defaults(rcParams):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["STIXGeneral"]
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "serif"
    rcParams["mathtext.sf"] = "serif"
    rcParams['mathtext.it'] = 'serif:italic'
    return rcParams
