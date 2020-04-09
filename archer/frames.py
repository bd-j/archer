#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Coordinate frames used for calculating quantities
"""

import numpy as np
import astropy.coordinates as coord
import astropy.units as u

from astropy.coordinates import galactocentric_frame_defaults
_ = galactocentric_frame_defaults.set('v4.0')

# --------------
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

# -----------
# DL17

# uses the much higher V_solar (~250 km/s as opposed to 220)
v_sun_dl17 = coord.CartesianDifferential([11.1, 237+12.24, 7.25] * u.km / u.s)
gc_frame_dl17 = coord.Galactocentric(galcen_distance=8.0 * u.kpc,
                                     z_sun=0 * u.pc,
                                     galcen_v_sun=v_sun_dl17)

gc_frame_dl17_nbody = coord.Galactocentric(galcen_distance=7.0 * u.kpc,
                                           z_sun=3 * u.pc,
                                           galcen_v_sun=v_sun_dl17)


# note DL17 give ell, b; these are converted to ra, dec
from astropy.coordinates import ICRS, Galactic, FK5
_sgr_dl17_galactic = coord.SkyCoord(5.6 * u.deg, -14.2 * u.deg, frame=Galactic)
_sgr_dl17_equatorial = _sgr_dl17_galactic.transform_to(FK5)
ra, dec = _sgr_dl17_equatorial.ra.value, _sgr_dl17_equatorial.dec.value
sgr_dl17 = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                          distance=25.2 * u.kpc,
                          pm_ra_cosdec=-2.95 * u.mas / u.yr,
                          pm_dec=-1.19 * u.mas / u.yr,
                          radial_velocity=140 * u.km / u.s)
# note that
# this results in GSR velocities different by ~4 km/s than DL17,
# though with a total velocity within 1 km/s (332 vs 333)

# ---------------
# Gaia based
sgr_fritz18 = coord.SkyCoord(ra=283.7629 * u.deg, dec=-30.4783 * u.deg,
                             distance=26.6 * u.kpc,
                             pm_ra_cosdec=-2.736 * u.mas / u.yr,
                             pm_dec=-1.357 * u.mas / u.yr,
                             radial_velocity=140 * u.km / u.s)
