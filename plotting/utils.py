#!/usr/bin/python

import numpy as np
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

__all__ = ["h3_quiver", "lm_quiver",
           "read_lm", "read_segue",
           "get_values",
           "get_sgr", "gsr_to_rv",
           "get_Lsgr", "compute_Lstar", "angle_to_sgr",
           "v_sun_law10", "gc_frame_law10", "sgr_law10", "sgr_fritz18"
           ]


vmap = {"x": "u",
        "y": "v",
        "z": "w"}


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



lmcols = {"ra": "ra", "dec": "dec",
          "pmra": "mua", "pmdec": "mud",
          "dist": "dist"}
rcatcols = {"ra": "RA", "dec": "DEC",
            "pmra": "GaiaDR2_pmra", "pmdec": "GaiaDR2_pmdec",
            "dist": "dist_adpt"}


def get_values(cat, sgr=sgr_law10, frame=gc_frame_law10):
    if "mua" in cat.dtype.names:
        # LM10
        cols = lmcols
        vlos = gsr_to_rv(cat["v"], cat["ra"], cat["dec"], cat["dist"], 
                         gc_frame=frame)
        vlos = vlos.value
    else:
        cols = rcatcols
        vlos = cat["Vrad"]

    Lsgr = get_Lsgr(sgr, gc_frame=frame)
    lstar = compute_Lstar(cat[cols["ra"]], cat[cols["dec"]], cat[cols["dist"]],
                          cat[cols["pmra"]], cat[cols["pmdec"]], vlos,
                          gc_frame=frame)
    phi_sgr, lsgr = angle_to_sgr(Lsgr, lstar)
    lx, ly, lz = cat["Lx"], cat["Ly"], cat["Lz"]
    etot = cat["E_tot_pot2"]

    return etot, lx, ly, lz, phi_sgr, lsgr


def h3_quiver(cat, zz, vtot=1.0, show="xy", ax=None, scale=20,
              cmap="viridis", **quiver_kwargs):
    x, y = [cat["{}_gal".format(s.upper())] for s in show]
    vx, vy = [cat["V{}_gal".format(s)] for s in show]
    if zz is not None:
        cb = ax.quiver(x, y, vx / vtot, vy / vtot, zz,
                       angles="xy", pivot="mid", cmap=cmap,
                       scale_units="height", scale=scale, **quiver_kwargs)
    else:
        cb = ax.quiver(x, y, vx / vtot, vy / vtot,
                       angles="xy", pivot="mid", cmap=cmap,
                       scale_units="height", scale=scale, **quiver_kwargs)
        
    return cb


def lm_quiver(cat, zz, vtot=1.0, show="xy", ax=None, scale=20,
              cmap="viridis", **quiver_kwargs):
    x, y = [cat["{}gc".format(s)] for s in show]
    vx, vy = [cat[vmap[s]] for s in show]
    cb = ax.quiver(x, y, vx / vtot, vy / vtot, zz,
                   angles="xy", pivot="mid", cmap=cmap,
                   scale_units="height", scale=scale, **quiver_kwargs)

    return cb


def overplot_clump(axes, subcat, clumpcolor="maroon"):
    axes[1, -1].plot(subcat["X_gal"], subcat["Y_gal"], 'o',
                     alpha=0.5, color=clumpcolor)
    axes[0, -1].plot(subcat["X_gal"], subcat["Z_gal"], 'o',
                     alpha=0.5, color=clumpcolor)
    return axes


def read_lm(lmfile):

    try:
        lm = fits.getdata(lmfile)
        # switch colums
        for a in "xyz":
            lm["{}gc".format(a)] = lm["{}_gal".format(a.upper())]
            lm[vmap[a]] = lm["V{}_gal".format(a)]

    except(OSError):
        with open(lmfile, "r") as f:
            cols = f.readline().split()
            dt = np.dtype([(c, np.float) for c in cols])
            lm = np.genfromtxt(lmfile, skip_header=1, dtype=dt)
        # Make right handed
        lm["xgc"] = -lm["xgc"]
        lm["u"] = -lm["u"]

    return lm


def read_segue(seguefile, dtype):

    name_map = {
                # quantities
                "RA": "ra",
                "DEC": "dec",
                "dist_adpt": "Dist",
                "GaiaDR2_pmra": "gaia.pmra",
                "GaiaDR2_pmdec": "gaia.pmdec",
                "Vrad": "HRV",
                "std_Vrad": "e_HRV",
                "feh": "FeH",
                "SDSS_R": "rmag",
                } 

    segin = fits.getdata(seguefile)
    nrow = len(segin)
    segout = np.zeros(nrow, dtype=dtype)

    for c in segout.dtype.names:
        if c in segin.dtype.names:
            segout[c] = segin[c]
        elif c in name_map:
            segout[c] = segin[name_map[c]]

    segout["SNR"] = 20.0
    sgr = get_sgr(segout)
    segout["Sgr_l"] = np.mod(sgr.Lambda.value, 360.)
    segout["Sgr_b"] = sgr.Beta.value

    return segout


def get_sgr(cat):
    import astropy.units as u
    import astropy.coordinates as coord
    import gala.coordinates as gc
    ceq = coord.ICRS(ra=cat['RA']*u.deg, dec=cat['DEC']*u.deg,
                     distance=cat["dist_adpt"] * u.kpc,
                     pm_ra_cosdec=cat['GaiaDR2_pmra']*u.mas/u.yr, pm_dec=cat['GaiaDR2_pmdec']*u.mas/u.yr,
                     radial_velocity=cat['Vrad']*u.km/u.s)
    sgr = ceq.transform_to(gc.Sagittarius)
    return sgr


def get_Lsgr(sgr_icrs, gc_frame=coord.Galactocentric()):

    sgr_gc = sgr_icrs.transform_to(gc_frame)
    # because cross doesn't work on sgr_gc.cartesian & sgr_gc.velocity
    xx = np.array([getattr(sgr_gc, a).to("kpc").value for a in "xyz"])
    p = np.array([getattr(sgr_gc, "v_{}".format(a)).to("km/s").value for a in "xyz"])
    # units are kpc * km/s
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
    """ Returns the cosine of the angle between Lsgr and Lstar:

        Lstar: shape (nstar, 3)
    """
    v1u = Lsgr / np.linalg.norm(Lsgr)
    v2u = Lstar.T / np.linalg.norm(Lstar, axis=-1)
    costheta = np.dot(v1u, v2u)
    projection = np.linalg.norm(Lstar, axis=-1) * costheta
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



