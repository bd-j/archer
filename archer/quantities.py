#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code for calculating quantities from basic observables or mock data inputs
"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from gala.coordinates import Sagittarius, reflex_correct

from astropy.coordinates import galactocentric_frame_defaults
_ = galactocentric_frame_defaults.set('v4.0')


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


def compute_energy(cat, potential):
    pos = np.array([cat['X_gal'], cat['Y_gal'], cat['Z_gal']]) * u.kpc
    vel = np.array([cat['Vx_gal'], cat['Vy_gal'], cat['Vz_gal']]) * u.km / u.s
    w = gd.PhaseSpacePosition(pos=pos, vel=vel.to(u.kpc/u.Myr))
    orbit = compute_orbit(w, potential)
    cat["ekin"] = w.kinetic_energy()
    cat["etot"] = orbit.energy()[0]
    cat["eccen"] = orbit.eccentricity()
    cat["zmax"] = orbit.zmax(func=np.max)
    cat["rapo"] = orbit.apocenter(func=np.max)
    cat["rperi"] = orbit.pericenter(func=np.min)
    cat["period"] = orbit.estimate_period()
    return cat


def compute_vtan(cat):
    vtan = 4.74 * np.hypot(cat["pmra"], cat["pmdec"]) * cat["dist"] 
    return vtan


def compute_orbit(w, potential):
    raise(NotImplementedError)


def reflex_uncorrect(cat=None, ceq=None, gc_frame=coord.Galactocentric()):
    """The DL17 proper motions are reflex corrected (i.e GSR). This function
    will put them back into the LSR.

    Parameters
    ----------
    cat : numpy structured array
        With reflex corrected quantities in the pmra, pmdec, and vrad fields

    gc_frame : `~astropy.coordinates.Galactocentric` (optional)
        To change properties of the Galactocentric frame, like the height of the
        sun above the midplane, or the velocity of the sun in a Galactocentric
        intertial frame, set arguments of the
        `~astropy.coordinates.Galactocentric` object and pass in to this
        function with your coordinates.

    Returns
    -------
    cat : numpy structured array
        Same as inoput, but with solar motion included in the velocity fields
        (i.e., having pm and velocity as would be observed)

    """
    if cat is not None:
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
    if cat is not None:
        cat["pmra"] = uncorr.pm_ra_cosdec.value
        cat["pmdec"] = uncorr.pm_dec.value
        cat["vrad"] = uncorr.radial_velocity.value
        return cat
    else:
        return uncorr


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


def estar_to_e(estar):
    """Convert from a uniform distribution (estar, effectively the value of the
    CDF) to actual energies assuming the DF of a Plummer sphere: p(E)~E^{5-3/2}
    
    Also, get s_max, where E = psi + v^2/2; v = 0, psi = 1/sqrt(1 + s^2)
    """
    # CDF is ~ E^{9/2}, so E ~ CDF^{2/9}
    e = (1-estar)**(2./9.)
    ssq = 1/e*2 - 1
    s = np.sqrt(ssq)
    return e, s
