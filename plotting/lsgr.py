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

from utils import read_lm, lm_quiver, h3_quiver

v_sun_law10 = coord.CartesianDifferential([11.1, 220, 7.25]*u.km/u.s)
gc_frame_law10 = coord.Galactocentric(galcen_distance=8.0*u.kpc,
                                      z_sun=0*u.pc,
                                      galcen_v_sun=v_sun_law10)

sgr_law10 = coord.SkyCoord(ra=283.7629*u.deg, dec=-30.4783*u.deg,
                           distance=28.0*u.kpc,
                           pm_ra_cosdec=-2.45*u.mas/u.yr, 
                           pm_dec=-1.30*u.mas/u.yr,
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
    projection =  np.linalg.norm(Lstar, axis=-1) * costheta
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
    lmTheta, lmProj = angle_to_sgr(Lsgr_lm10, lstar_lm10)
    
    # --- H3 ----
    rcat = fits.getdata(rcatfile)
    Lsgr_h3 = get_Lsgr(sgr_law10, gc_frame=gc_frame_law10)
    lstar_h3 = compute_Lstar(rcat["RA"], rcat["DEC"], rcat["dist_adpt"], 
                             rcat["GaiaDR2_pmra"], rcat["GaiaDR2_pmdec"], 
                             rcat["Vrad"],
                             gc_frame=gc_frame_law10)
    h3Theta, h3Proj = angle_to_sgr(Lsgr_h3, lstar_h3)


    # --- Basic selections ---
    vtot_lm = np.sqrt(lm["u"]**2 + lm["v"]**2 + lm["w"]**2)
    vtot = np.sqrt(rcat["Vx_gal"]**2 + rcat["Vy_gal"]**2 + rcat["Vz_gal"]**2)
    feh = np.clip(rcat["feh"], -2.0, 0.1)
    good = ((rcat["logg"] < 3.5) & np.isfinite(rcat["Z_gal"]) &
            (rcat["FLAG"] == 0) & np.isfinite(vtot) & (vtot < 3000))
    sgr = np.abs(rcat["Sgr_B"]) < 40
    far = rcat["R_gal"] > 20
    etot = rcat["E_tot_pot1"]
    theta_thresh = 0.75
    
    
    np.random.seed(101)
    linds = np.random.choice(len(lm), size=int(0.1 * len(lm)))
    lmsel = np.zeros(len(lm), dtype=bool)
    lmsel[linds] = True
    lmhsel = (lm["in_h3"] == 1) & (np.random.uniform(0, 1, len(lm)) < 0.5) #& (lmTheta > theta_thresh)
    
    import matplotlib.pyplot as pl
    dfig, dax = pl.subplots()
    peri = np.unique(lm["Pcol"])
    for p in peri:
        psel = lmhsel & (lm["Pcol"] == p)
        dax.hist(lmTheta[psel], bins=100, range=(-1, 1), 
                 alpha=0.5, label="LM10 peri #{}".format(peri.max()-p))
    hfig, hax = pl.subplots()
    hax.hist(h3Theta[good], bins=100, range=(-1, 1), alpha=0.3, color="maroon", label="H3")
    [ax.set_xlabel(r"$\cos \theta_{Sgr}$") for ax in [hax, dax]]
    [ax.legend() for ax in [hax, dax]]

    dfig.savefig("figures/theta_sgr_dist.lm10{}.png".format(noisiness))
    hfig.savefig("figures/theta_sgr_dist.h3v{}.png".format(rcat_vers))

    lcatz = [lm["Pcol"], lm["Pcol"]]
    rcatz = [feh, feh]
    ascale = 25
    nrow, ncol = 2, 3
    projections = ["xz", "xy"]

    
    sel = good & (h3Theta > theta_thresh)
    fig, axes = pl.subplots(nrow, ncol, sharex=True, sharey="row", figsize=(24, 12.5))
    cbl = [lm_quiver(lm[lmsel], z[lmsel], vtot=vtot_lm[lmsel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, axes[:, 0], lcatz)]
    cblh = [lm_quiver(lm[lmhsel], z[lmhsel], vtot=vtot_lm[lmhsel], show=s, ax=ax, scale=ascale)
            for s, ax, z in zip(projections, axes[:, 1], lcatz)]
    cbh = [h3_quiver(rcat[sel], z[sel], vtot=vtot[sel], show=s, ax=ax, scale=ascale)
           for s, ax, z in zip(projections, axes[:, 2], rcatz)]

    axes[0, 0].set_xlim(-70, 40)
    axes[0, 0].set_ylim(-80, 80)
    axes[1, 0].set_ylim(-30, 40)
    [ax.xaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[0, :]]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) for ax in axes[:, 1:].flat]

    axes[0, 0].set_title("LM10")
    axes[0, 1].set_title("LM10xH3")
    axes[0, 2].set_title("H3")
    for i in range(nrow):
        xl, yl = projections[i]
        for j in range(ncol):
            axes[i, j].set_xlabel(xl.upper())
            axes[i, j].set_ylabel(yl.upper())

    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())

    fig.subplots_adjust(hspace=0.15, wspace=0.2)

    fig.savefig("figures/sgr_lm10_h3v{}_{}.cutTheta_sgr.png".format(rcat_vers, noisiness), dpi=150)