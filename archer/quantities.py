#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code for calculating quantities from basic observables or mock data inputs
"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from gala.coordinates import Sagittarius, reflex_correct


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


def ref_corr(cat, gc_frame=coord.Galactocentric()):
    
    ceq = coord.ICRS(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg,
                     distance=cat["dist"] * u.kpc,
                     pm_ra_cosdec=cat['pmra'] * u.mas / u.yr,
                     pm_dec=cat['pmdec'] * u.mas / u.yr,
                     radial_velocity=cat['vrad'] * u.km / u.s)
    reflex_correct(ceq)
    cat["pmra"] = ceq.pm_ra_cosdec
    cat["pmdec"] = ceq.pm_dec
    cat["vrad"] = ceq.radial_velocity
    return cat



def my_reflex_correct(coords, galactocentric_frame=None):
    """Correct the input Astropy coordinate object for solar reflex motion.

    The input coordinate instance must have distance and radial velocity information. If the radial velocity is not known, fill the

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The Astropy coordinate object with position and velocity information.
    galactocentric_frame : `~astropy.coordinates.Galactocentric` (optional)
        To change properties of the Galactocentric frame, like the height of the
        sun above the midplane, or the velocity of the sun in a Galactocentric
        intertial frame, set arguments of the
        `~astropy.coordinates.Galactocentric` object and pass in to this
        function with your coordinates.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The coordinates in the same frame as input, but with solar motion
        removed.

    """
    c = coord.SkyCoord(coords)

    # If not specified, use the Astropy default Galactocentric frame
    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric()

    v_sun = galactocentric_frame.galcen_v_sun

    observed = c.transform_to(galactocentric_frame)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    fr = galactocentric_frame.realize_frame(rep).transform_to(c.frame)
    return coord.SkyCoord(fr)



def reflex_uncorrect(cat, gc_frame=coord.Galactocentric()):
    """The DL17 proper motions are reflex corrected (i.e GSR).
    This function will put them back into the LSR.

    Parameters
    ----------
    cat : numpy structured array
        
    gc_frame : `~astropy.coordinates.Galactocentric` (optional)
        To change properties of the Galactocentric frame, like the height of the
        sun above the midplane, or the velocity of the sun in a Galactocentric
        intertial frame, set arguments of the
        `~astropy.coordinates.Galactocentric` object and pass in to this
        function with your coordinates.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The coordinates in the same frame as input, but with solar motion
        included (i.e., if the input is ICRS or FK5 or so, the resulting
        coordinates would have pm and velocity as would be observed)

    """
    
    ceq = coord.ICRS(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg,
                     distance=cat["dist"] * u.kpc,
                     pm_ra_cosdec=cat['pmra'] * u.mas / u.yr,
                     pm_dec=cat['pmdec'] * u.mas / u.yr,
                     radial_velocity=cat['vrad'] * u.km / u.s)


    c = coord.SkyCoord(ceq)
    # Transform to galctocentric
    corr = c.transform_to(gc_frame)
    # subtract solar velocity
    v_sun = gc_frame.galcen_v_sun
    rep = corr.cartesian.without_differentials()
    rep = rep.with_differentials(corr.cartesian.differentials['s'] - v_sun)
    fr = gc_frame.realize_frame(rep).transform_to(c.frame)
    uncorr = coord.SkyCoord(fr)
    cat["pmra"] = uncorr.pm_ra_cosdec.value
    cat["pmdec"] = uncorr.pm_dec.value
    cat["vrad"] = uncorr.radial_velocity.value
    return cat


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


