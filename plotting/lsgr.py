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
    extra = (rcat["Vrot"] < 5) & (rcat["SNR"] > 3)
    good = good & extra
    sgr = np.abs(rcat["Sgr_B"]) < 40
    far = rcat["R_gal"] > 10
    etot = rcat["E_tot_pot2"]
    theta_thresh = 0.8
    aligned = (h3Theta > theta_thresh)
    clump = (
             (h3Proj > 1000) & (h3Proj < 4000) &
             (etot > -175000) & (etot < -140000) &
             #(h3Proj > 5000) & (h3Proj < 10000) & 
             #(etot > -88500) & (etot < -75500) &
             aligned & good
             )
    
    
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
    hax.hist(h3Theta[good & far], bins=100, range=(-1, 1), alpha=0.3, color="maroon", label="H3")
    [ax.set_xlabel(r"$\cos \phi_{Sgr}$") for ax in [hax, dax]]
    [ax.legend() for ax in [hax, dax]]

    dfig.savefig("figures/theta_sgr_dist.lm10{}.png".format(noisiness))
    hfig.savefig("figures/theta_sgr_dist.h3v{}.png".format(rcat_vers))

    lcatz = [lm["Pcol"], lm["Pcol"]]
    rcatz = [feh, feh]
    ascale = 25
    nrow, ncol = 2, 3
    projections = ["xz", "xy"]

    
    sel = good & aligned & far
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

    clumpcolor = "orange"
    axes[1, -1].plot(rcat["X_gal"][clump], rcat["Y_gal"][clump], 'o', 
                     alpha=0.5, color=clumpcolor)
    axes[0, -1].plot(rcat["X_gal"][clump], rcat["Z_gal"][clump], 'o', 
                     alpha=0.5, color=clumpcolor)

    #for i in range(2):
    #    c = fig.colorbar(cbh[i], ax=axes[i,:])
    #    c.set_label("yz"[i].upper())

    fig.subplots_adjust(hspace=0.15, wspace=0.2)
    fig.savefig("figures/sgr_lm10_h3v{}_{}.cutTheta_sgr.png".format(rcat_vers, noisiness), dpi=150)
    
    
    # --- Bifurcation ---
    stream = good & (rcat["Sgr_l"] > 200) & (rcat["Sgr_l"] < 300)
    A = stream & (rcat["Sgr_b"] > 0)
    B = stream & (rcat["Sgr_b"] < 0)
    
    
    # --- E-Lsgr ---
    efig, eaxes = pl.subplots(1, 2, sharey=True)
    eax = eaxes[1]
    eax.plot(h3Proj[good & ~sel], rcat[good & ~sel]["E_tot_pot2"], 'o', alpha=0.4)
    eax.plot(h3Proj[sel], rcat[sel]["E_tot_pot2"], 'o', alpha=0.4)
    eax = eaxes[0]
    eax.scatter(lmProj[lmhsel], lm[lmhsel]["E_tot_pot2"], c=lm[lmhsel]["Pcol"], marker='o', alpha=0.4)
    
    [a.set_ylim(-2e5, -5e4) for a in eaxes]
    [a.set_xlim(-1e4, 2e4) for a in eaxes]
    
    # --- Vgsr lambda ----

    lfig, laxes = pl.subplots(1, 3, sharex=True, sharey=True,
                            figsize=(16, 4))

    lax = laxes[0]
    lbm = lax.scatter(lm[lmsel]["lambda"], lm[lmsel]["V_gsr"], 
                    c=lm[lmsel]["Pcol"], marker=".", s=1,
                    vmin=-2, vmax=3, cmap="viridis")
    lax = laxes[1]
    lbm = lax.scatter(lm[lmhsel]["lambda"], lm[lmhsel]["V_gsr"], 
                    c=lm[lmhsel]["Pcol"], marker="o", alpha=0.7, s=4,
                    vmin=-2, vmax=3, cmap="viridis")

    lax = laxes[2]
    vsel = good  & far
    lbh = lax.scatter(rcat[sel]["sgr_l"], rcat["V_gsr"][sel], 
                    c=rcat["dist_adpt"][sel], marker="o", alpha=0.7, s=4,
                    vmin=0, vmax=100, cmap="viridis")

    lax.set_ylim(-350, 350)
    laxes[0].set_title("LM10")
    laxes[1].set_title("LM10xH3")
    laxes[2].set_title("H3")

    [ax.set_xlabel(r"$\Lambda_{Sgr}$") for ax in laxes]
    [ax.set_ylabel(r"$V_{GSR}$") for ax in laxes]
    [ax.yaxis.set_tick_params(which='both', labelbottom=True) 
     for ax in laxes[1:]]
    [ax.invert_xaxis() for ax in laxes]

    lfig.colorbar(lbh, ax=laxes)
    
    
    # --- FeH ---
    
    trail = (good & aligned & 
             (rcat["sgr_l"] < 150) & (rcat["V_gsr"] < 0) &
             (etot > -150000))
    
    lead = (good & aligned & 
             (rcat["sgr_l"] > 200) & (rcat["V_gsr"] < 25) & (rcat["V_gsr"] > -140) &
             (etot > -150000))
    
    zfig, zax = pl.subplots()
    zax.hist(feh[trail], bins=30, alpha=0.5, label="Trailing")
    zax.hist(feh[lead], bins=30, alpha=0.5, label="Leading")
    zax.set_xlim(-3, 0.0)
    zax.legend()
    zax.set_xlabel(r"[Fe/H]")