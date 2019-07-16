#!/usr/bin/python

"""Script to calculate the angular offset between H3 stars and the 
Sagittarius dwarf galaxy angular momentum vector. Do it for mocks as well.
"""

import numpy as np
import gala

from astropy.io import fits
import astropy.units as u
import astropy.coordinates as coord
import gala.coordinates as gc

from utils import read_lm

v_sun_law10 = coord.CartesianDifferential([11.1, 220, 7.25]*u.km/u.s)
gc_frame_law10 = coord.Galactocentric(galcen_distance=8.0*u.kpc,
                                      z_sun=0*u.pc,
                                      galcen_v_sun=v_sun_law10)

sgr_law10 = coord.SkyCoord(ra=283.7629*u.deg, dec=-30.4783*u.deg,
                           distance=28.0*u.kpc,
                           pm_ra_cosdec=-2.16*u.mas/u.yr, 
                           pm_dec=1.73*u.mas/u.yr,
                           radial_velocity=171*u.km/u.s)

sgr_fritz18 = coord.SkyCoord(ra=283.7629*u.deg, dec=-30.4783*u.deg,
                             distance=26.6*u.kpc,
                             pm_ra_cosdec=-2.736*u.mas/u.yr, 
                             pm_dec=-1.357*u.mas/u.yr,
                             radial_velocity=140*u.km/u.s)


def get_Lsgr(sgr_icrs, gc_frame=coord.Galactocentric()):

    sgr_gc = sgr_icrs.transform_to(gc_frame)
    xx = np.array([getattr(sgr_gc, a).to("kpc").value for a in "xyz"])
    p = np.array([getattr(sgr_gc, "v_{}".format(a)).to("km/s").value for a in "xyz"])
    L = np.cross(xx, p)
    return L


def compute_Lstar(ra, dec, distance, pmra, pmdec, vlos,
                  gc_frame=coord.Galactocentric()):
    
    ceq = coord.ICRS(ra=ra*u.deg, dec=dec*u.deg,
                     distance=distance * u.kpc,
                     pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr,
                     radial_velocity=vlos*u.km/u.s)
    
    gc = ceq.transform_to(gc_frame)
    xx = np.array([getattr(gc, a).to("kpc").value for a in "xyz"]).T
    p = np.array([getattr(gc, "v_{}".format(a)).to("km/s").value for a in "xyz"]).T
    Lstar = np.cross(xx, p)
    
    return Lstar


def angle_to_sgr(Lsgr, Lstar):
    
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
        
        
        Lstar: shape (nstar, 3)
    """
    v1u = Lsgr / np.linalg.norm(Lsgr)
    v2u = Lstar.T / np.linalg.norm(Lstar, axis=-1)
    costheta = np.dot(v1u, v2u)
    projection =  np.linalg.norm(Lstar) * costheta
    return costheta, projection


def gsr_to_rv(vgsr, ra, dec, dist, gc_frame=coord.Galactocentric()):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

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
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    
    c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc)
    v_sun = gc_frame.galcen_v_sun.to_cartesian()

    gal = c.transform_to(gc_frame)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return vgsr*u.km/u.s - v_proj



if __name__ == "__main__":
    
    lmocks = ["../data/mocks/LM10/SgrTriax_DYN.dat",
              "../data/mocks/LM10/LM10_h3_noiseless_v1.fits",
              "../data/mocks/LM10/LM10_h3_noisy_v1.fits"]
    rcat_vers = "1.4"
    rcatfile = "../data/catalogs/rcat_V{}_MSG.fits".format(rcat_vers)
    lmockfile, noisiness = lmocks[1], "noiseless"

    # --- L & M 2010 model ---
    lm = read_lm(lmockfile)
    Lsgr_lm10 = get_Lsgr(sgr_law10, gc_frame=gc_frame_law10)
    vlos = gsr_to_rv(lm["v"], lm["ra"], lm["dec"], lm["dist"])
    
    lstar_lm10 = compute_Lstar(lm["ra"], lm["dec"], lm["dist"], 
                               lm["mua"], lm["mud"], vlos.value,
                               gc_frame=gc_frame_law10)
    lmTheta, lmProj = angle_to_sgr(Lsgr, lstar_lm10)
    
    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    Lsgr_h3 = get_Lsgr(sgr_law10, gc_frame=gc_frame_law10)
    lstar_h3 = compute_Lstar(rcat["RA"], rcat["DEC"], rcat["dist_adpt"], 
                             rcat["GaiaDR2_pmra"], rcat["GaiaDR2_pmdec"], 
                             rcat["Vrad"],
                             gc_frame=gc_frame_law10)
    h3Theta, h3Proj = angle_to_sgr(Lsgr, lstar_h3)



    # --- Basic selections ---
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
    feh = np.clip(rcat["feh"], -2.0, 0.1)
    good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
            (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
    sgr = np.abs(rcat["Sgr_B"]) < 40
    far = rcat["R_gal"] > 20
    etot = rcat["E_tot_pot1"]
    
    import matplotlib.pyplot as pl
    fig, dax = pl.subplots()
    dax.hist(lmTheta, bins=100, range=(-1, 1))
    dax.hist(h3Theta[good], bins=100, range=(-1, 1))

