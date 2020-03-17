#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from argparse import Namespace, ArgumentParser
import astropy.coordinates

epath = os.path.expandvars
pjoin = os.path.join

parser = ArgumentParser()

# --- Figures ---
parser.add_argument("--figure_extension", type=str,
                    default="png")
parser.add_argument("--figure_dir", type=str,
                    default="figures/")
parser.add_argument("--figure_dpi", type=int, default=450)
parser.add_argument("--savefig", action="store_true")


# --- catalog versions ---
parser.add_argument("--rcat_vers", type=str, default="2_0")

parser.add_argument("--add_noise", action="store_true")
parser.add_argument("--segue_cat", action="store_true")
parser.add_argument("--data_dir", type=str,
                    default=epath("$HOME/Projects/archer/data/"))

parser.add_argument("--lm10_file", type=str,
                    default="SgrTriax.fits")
parser.add_argument("--dl17_file", type=str,
                    default="DL17_all_orig.fits")
parser.add_argument("--r18_file", type=str,
                    default="R18_noiseless_v5.fits")
parser.add_argument("--segue_file", type=str,
                    default="ksegue_gaia_v5.fits")
parser.add_argument("-pcat_file", type=str,
                    default="pcat.fits")

def rectify_config(config):
    
    config.lm10_file = pjoin(config.data_dir, "mocks", "LM10", config.lm10_file)
    config.dl17_file = pjoin(config.data_dir, "mocks", "DL17", config.dl17_file)
    config.r18_file = pjoin(config.data_dir, "mocks", "R18", config.r18_file)
    config.segue_file = pjoin(config.data_dir, "catalogs", config.segue_file)

    fn = "rcat_V{}_MSG.fits".format(config.rcat_vers.replace("_", "."))
    config.rcat_file = pjoin(config.data_dir, "catalogs", fn)
    config.pcat_file = pjoin(config.data_dir, "catalogs", config.pcat_file)

    config.gc_frame = astropy.coordinates.Galactocentric()

    return config


def plot_defaults(rcParams):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["STIXGeneral"]
    rcParams["font.size"] = 12
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "serif"
    rcParams["mathtext.sf"] = "serif"
    rcParams['mathtext.it'] = 'serif:italic'
    return rcParams
