#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Coordinate frames used for calculating quantities
"""

import numpy as np
import astropy.coordinates as coord
import astropy.units as u

# Law 10
v_sun_law10 = coord.CartesianDifferential([11.1, 220, 7.25] * u.km / u.s)
gc_frame_law10 = coord.Galactocentric(galcen_distance=8.0 * u.kpc,
                                      z_sun=0 * u.pc,
                                      galcen_v_sun=v_sun_law10)

sgr_law10 = coord.SkyCoord(ra=283.7629 * u.deg, dec=-30.4783 * u.deg,
                           distance=28.0 * u.kpc,
                           pm_ra_cosdec=-2.45 * u.mas / u.yr,
                           pm_dec=-1.30 * u.mas / u.yr,
                           radial_velocity=171 * u.km / u.s)

sgr_fritz18 = coord.SkyCoord(ra=283.7629 * u.deg, dec=-30.4783 * u.deg,
                             distance=26.6 * u.kpc,
                             pm_ra_cosdec=-2.736 * u.mas / u.yr,
                             pm_dec=-1.357 * u.mas / u.yr,
                             radial_velocity=140 * u.km / u.s)
