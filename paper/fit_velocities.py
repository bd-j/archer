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

    config = rectify_config(parser.parse_args())

    rcat = fits.getdata(config.rcat_file)
    rcat_r = rectify(homogenize(rcat, "RCAT"), config.gc_frame)
    
    # selection
    good = (rcat["FLAG"] == 0) & (rcat["SNR"] > 3) & (rcat["logg"] < 3.5)
    sgr = rcat_r["ly"] < (-0.3 * rcat_r["lz"] - 0.25)
    sgr = sgr & (rcat_r["vgsr"] < 100) & (rcat_r["vgsr"] > -300)

    # --- Smaht fit ---
    # Set Priors; each prior is lower & upper of shape (2, norder)
    alpha_range = [(250, 450), (-10, -5), (-0.05, 0.05)]
    beta_range = [(-150, 0), (-1.0, 3), (-0.05, 0.05)]
    pout_range = np.array([0, 0.15])
    trail_model = VelocityModel(alpha_range=np.array(alpha_range).T,
                                beta_range=np.array(beta_range).T,
                                pout_range=pout_range)

    alpha_range = [(-500, 900), (-10, 0.1), (-0.05, 0.05)]
    beta_range = [(-300, 0), (0., 1.0)]
    pout_range = np.array([0.1, 0.5])
    lead_model = VelocityModel(alpha_range=np.array(alpha_range).T,
                                beta_range=np.array(beta_range).T,
                                pout_range=pout_range)

    #sys.exit()
    
    from dynesty import DynamicNestedSampler as Sampler

    if True:
        # Trailing
        selection = good & sgr & (rcat_r["lambda"] < 175)
        lam = rcat_r["lambda"][selection]
        vgsr = rcat_r["vgsr"][selection]
        trail_model.set_data(lam, vgsr)
        dsampler = Sampler(trail_model.lnprob, trail_model.prior_transform, trail_model.ndim)
        dsampler.run_nested()
        trail_results = dsampler.results
        write_to_h5(trail_results, trail_model, "h3_trailing_fit.h5")

    if True:
        # Leading
        selection = good & sgr & (rcat_r["lambda"] > 175)
        lam = rcat_r["lambda"][selection]
        vgsr = rcat_r["vgsr"][selection]
        lead_model.set_data(lam, vgsr)
        dsampler = Sampler(lead_model.lnprob, lead_model.prior_transform, lead_model.ndim)
        dsampler.run_nested()
        lead_results = dsampler.results
        write_to_h5(lead_results, lead_model, "h3_leading_fit.h5")
