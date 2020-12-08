#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from argparse import Namespace, ArgumentParser
import astropy.coordinates
from astropy.coordinates import galactocentric_frame_defaults
_ = galactocentric_frame_defaults.set('v4.0')

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


# --- data location ---
parser.add_argument("--data_dir", type=str,
                    default=epath("$HOME/Projects/archer/data/"))

# --- mock realism ---
parser.add_argument("--fractional_distance_error", type=float,
                    default=0.0)
parser.add_argument("--noisify_pms", action="store_true")
parser.add_argument("--mag_cut", action="store_true")

# --- selection ---
parser.add_argument("--dly", type=float,
                    default=0.0, help="shift selection line by this amount in Ly")
parser.add_argument("--flx", type=float,
                    default=0.9, help="allowed |Lx| as fraction of quadrature sum of Ly+Lz")
parser.add_argument("--max_rank", type=int,
                    default=2, help="maximum xfit rank to include in the selection")

# --- catalog versions ---
parser.add_argument("--rcat_vers", type=str, default="2_4")
parser.add_argument("--rcat_type", type=str, default="RCAT_KIN")
parser.add_argument("--gaia_vers", type=str, default="GAIADR2")

parser.add_argument("--use_segue", action="store_true")
parser.add_argument("--covar_dir", type=str,
                    default=epath("$HOME/Projects/archer/data/catalogs/covar"))

parser.add_argument("--lm10_file", type=str,
                    default="SgrTriax.fits")
parser.add_argument("--dl17_file", type=str,
                    default="DL17_all_orig.fits")
parser.add_argument("--r18_file", type=str,
                    default="R18_noiseless_v5.fits")
parser.add_argument("--segue_file", type=str,
                    default="ksegue_gaia_v5.fits")
parser.add_argument("--yang19_file", type=str,
                    default="yang19.fits")
parser.add_argument("--b19_file", type=str,
                    default="baumgardt19.fits")
parser.add_argument("--v19_file", type=str,
                    default="vasiliev19.fits")
parser.add_argument("-pcat_file", type=str,
                    default="pcat.fits")

# --- data files for SED making ---
parser.add_argument("--nnfile", type=str,
                    default="nnMIST_BC.h5")
parser.add_argument("--mistfile", type=str,
                    default="MIST_1.2_EEPtrk.h5")
parser.add_argument("--mistiso", type=str,
                    default="MIST_1.2_iso_vvcrit0.4.h5")



def rectify_config(config):

    config.lm10_file = pjoin(config.data_dir, "mocks", "LM10", config.lm10_file)
    config.dl17_file = pjoin(config.data_dir, "mocks", "DL17", config.dl17_file)
    config.r18_file = pjoin(config.data_dir, "mocks", "R18", config.r18_file)
    config.segue_file = pjoin(config.data_dir, "catalogs", config.segue_file)
    config.yang19_file = pjoin(config.data_dir, "catalogs", config.yang19_file)
    config.b19_file = pjoin(config.data_dir, "gcs", config.b19_file)
    config.v19_file = pjoin(config.data_dir, "gcs", config.v19_file)

    fn = "rcat_V{}_MSG.fits".format(config.rcat_vers.replace("_", "."))
    config.rcat_file = pjoin(config.data_dir, "catalogs", fn)
    config.pcat_file = pjoin(config.data_dir, "catalogs", config.pcat_file)

    config.nnfile = pjoin(config.data_dir, "sps", config.nnfile)
    config.mistfile = pjoin(config.data_dir, "sps", config.mistfile)
    config.mistiso = pjoin(config.data_dir, "sps", config.mistiso)

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
