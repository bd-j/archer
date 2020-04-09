#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import astropy.coordinates as coord

from archer.config import rectify_config, parser
from archer.catalogs import homogenize, rectify
from archer.frames import gc_frame_law10

from archer.fitting import VelocityModel, write_to_h5


if __name__ == "__main__":

    try:
        parser.add_argument("--fit_lm10", action="store_true")
        parser.add_argument("--fit_h3", action="store_true")
        parser.add_argument("--fit_leading", action="store_true")
        parser.add_argument("--fit_trailing", action="store_true")
    except:
        pass

    config = rectify_config(parser.parse_args())

    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
    pcat = fits.getdata(config.pcat_file)
    
    # lm10
    lm10 = fits.getdata(config.lm10_file)
    lm10_r = rectify(homogenize(lm10, "LM10", pcat=pcat),
                     gc_frame_law10)
 

    # selections
    from make_selection import rcat_select
    good, sgr = rcat_select(rcat, rcat_r)
    unbound = lm10["tub"] > 0

    # --- Smaht fit ---
    # Set Priors; each prior is lower & upper of shape (2, norder)
    alpha_range = [(250, 450), (-10, -5), (-0.05, 0.05)]
    beta_range = [(-160, 0), (-1.0, 4), (-0.05, 0.05)]
    pout_range = np.array([0, 0.15])
    trail_model = VelocityModel(alpha_range=np.array(alpha_range).T,
                                beta_range=np.array(beta_range).T,
                                pout_range=pout_range)

    alpha_range = [(-200, 1500), (-20, 0.1), (-0.05, 0.05)]
    beta_range = [(-500, 100), (-4., 5.0), (-0.03, 0.03)]
    pout_range = np.array([0.0, 0.5])
    lead_model = VelocityModel(alpha_range=np.array(alpha_range).T,
                                beta_range=np.array(beta_range).T,
                                pout_range=pout_range)

    #sys.exit()
    
    from dynesty import DynamicNestedSampler as Sampler

    print(config.fit_lm10)

    if config.fit_h3 & config.fit_trailing:
        # Trailing
        selection = good & sgr & (rcat_r["lambda"] < 175)
        lam = rcat_r["lambda"][selection]
        vgsr = rcat_r["vgsr"][selection]
        trail_model.set_data(lam, vgsr)
        dsampler = Sampler(trail_model.lnprob, trail_model.prior_transform, trail_model.ndim)
        dsampler.run_nested()
        trail_results = dsampler.results
        write_to_h5(trail_results, trail_model, "fits/h3_trailing_fit.h5")

    if config.fit_h3 & config.fit_leading:
        # Leading
        selection = good & sgr & (rcat_r["lambda"] > 175)
        lam = rcat_r["lambda"][selection]
        vgsr = rcat_r["vgsr"][selection]
        lead_model.set_data(lam, vgsr)
        dsampler = Sampler(lead_model.lnprob, lead_model.prior_transform, lead_model.ndim)
        dsampler.run_nested()
        lead_results = dsampler.results
        write_to_h5(lead_results, lead_model, "fits/h3_leading_fit.h5")

    if config.fit_lm10 & config.fit_trailing:
        # Trailing
        selection = (unbound & (lm10_r["lambda"] < 175) & (lm10_r["in_h3"] == 1.) &
                     (lm10["Lmflag"] == -1) & (lm10["Pcol"] < 3))
        lam = lm10_r["lambda"][selection]
        vgsr = lm10_r["vgsr"][selection]
        trail_model.set_data(lam, vgsr)
        dsampler = Sampler(trail_model.lnprob, trail_model.prior_transform, trail_model.ndim)
        dsampler.run_nested()
        trail_results = dsampler.results
        write_to_h5(trail_results, trail_model, "fits/lm10_trailing_fit.h5")

    if config.fit_lm10 & config.fit_leading:
        # Leading
        selection = (unbound & (lm10_r["lambda"] > 175) & (lm10_r["in_h3"] == 1.) &
                     (lm10["Lmflag"] == 1) & (lm10["Pcol"] < 3))
        lam = lm10_r["lambda"][selection]
        vgsr = lm10_r["vgsr"][selection]
        lead_model.set_data(lam, vgsr)
        dsampler = Sampler(lead_model.lnprob, lead_model.prior_transform, lead_model.ndim)
        dsampler.run_nested()
        lead_results = dsampler.results
        write_to_h5(lead_results, lead_model, "fits/lm10_leading_fit.h5")
