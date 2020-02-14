#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code for calculating quantities from basic observables or mock data inputs
"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from gala.coordinates import Sagittarius


def sgr_coords(cat, sgr_frame=Sagittarius):
    ceq = coord.ICRS(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg,
                     distance=cat["dist"] * u.kpc,
                     pm_ra_cosdec=cat['pmra'] * u.mas / u.yr,
                     pm_dec=cat['pmdec'] * u.mas / u.yr,
                     radial_velocity=cat['vrad'] * u.km / u.s)
    sgr = ceq.transform_to(sgr_frame)
    return sgr


def galactic_coords(cat, gc_frame=coord.Galactocentric()):
    ceq = coord.ICRS(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg,
                     distance=cat["dist"] * u.kpc,
                     pm_ra_cosdec=cat['pmra'] * u.mas / u.yr,
                     pm_dec=cat['pmdec'] * u.mas / u.yr,
                     radial_velocity=cat['vrad'] * u.km / u.s)
    gc = ceq.transform_to(gc_frame)
    return gc


def compute_lstar(cat, gc_frame=coord.Galactocentric()):
    gc = galactic_coords(cat, gc_frame)
    xx = np.array([getattr(gc, a).to("kpc").value for a in "xyz"]).T
    p = np.array([getattr(gc, "v_{}".format(a)).to("km/s").value
                  for a in "xyz"]).T
    Lstar = np.cross(xx, p)

    return Lstar, gc


def rv_to_gsr(cat, gc_frame=coord.Galactocentric()):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    c = coord.ICRS(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg,
                   radial_velocity=cat["vrad"] * u.km/u.s)

    v_sun = gc_frame.galcen_v_sun.to_cartesian()
    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj


def gsr_to_rv(cat, gc_frame=coord.Galactocentric()):
    """Transform a velocity in the Galactic Standard of Rest
    (GSR) to the barycentric radial velocity.

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity` (optional)
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    vrad : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    c = coord.ICRS(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg)
#                   distance=cat["dist"] * u.kpc)

    v_sun = gc_frame.galcen_v_sun.to_cartesian()
    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return cat["vgsr"] * u.km / u.s - v_proj


