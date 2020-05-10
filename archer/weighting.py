#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl


def kroupa_imf(minit):
    alpha = -2.3 * np.ones_like(minit)
    alpha[minit < 0.5] = -1.3
    alpha[minit < 0.08] = -0.3
    return alpha


def imf_weight(mass_array):
    delta_m = np.gradient(mass_array)
    wght = mass_array**(kroupa_imf(mass_array)) * delta_m
    # FIXME: this should normalize to a fixed mass interval,
    # not just the finite values, right?
    wght /= np.nansum(wght)
    return wght


def get_lf(iso, loga=10, feh=-1, afe=0):
    mags, params, _ = iso.get_seds(loga=loga, feh=feh, afe=afe,
                                   apply_corr=False, dist=10)
    params["eep"] = iso.eep_u.copy()
    weights = imf_weight(params["mini"])
    mags = dict(zip(iso.filters, mags.T))
    return mags, params, weights


def lf_observable_fraction(mags, wght, d_kpc, mask=True,
                           selection_band="PS_r", faint=18, bright=15,
                           wisecut=False):
    """Given an LF ([mag, weight] pair), compute corrections for a range of distances
    """
    mu = 5 * np.log10(d_kpc) + 10
    frac = np.zeros_like(mu)
    for i, m in enumerate(mu):
        mag = mags[selection_band] + m
        valid = (mag > bright) & (mag < faint) & mask
        if wisecut:
            w1 = mags["WISE_W1"] + m
            valid = valid & (w1 < 15.5)
        frac[i] = np.nansum(wght[valid])
    return frac


def mag_weight(dist, feh, iso, masker=None,
               bright=15, faint=18, selection_band="PS_r",
               lf_params=dict(afe=0, loga=10), wisecut=False):
    """Compute fraction of stars of a given feh and distance that made it into
    the magnitude selection (and other conditions given by `masker`), based on
    the LF for that metallicity. Takes about 50ms per object

    Parameters
    ----------
    dist : float
        Distance in kpc

    feh : float
        [Fe/H] of the target

    iso : seds.Isochrone instance
        Assumes that the mag selection band (i.e. PS_r) is the first element
        of the output magnitudes

    masker : callable, optional
        A function that takes a params structured array and a set of absolute
        magnitudes and returns a boolean array indicating whether each element
        (isochrone point) is a valid member

    bright : float (optional, default 15)
        Bright limit to have made it into the selection.

    faint : float (optional, default 18)
        Faint limit to have made it into the selection

    lf_params : dict
        Dictionary of parameters (e.g. loga, afe) for generating the isochrone & LF
    
    Returns
    -------
    mag_weight : float
        Fraction of stars at the supplied distance and metallcitiy that are
        within the given magnitude range and selected by `masker`.
    """
    band = selection_band
    assert (band in iso.filters)

    # Construct LF
    mu = 5 * np.log10(dist) + 10
    marr, params, _ = iso.get_seds(feh=feh, apply_corr=False, dist=10,
                                   **lf_params)
    params["eep"] = iso.eep_u.copy()
    weights = imf_weight(params["mini"])
    mags = dict(zip(iso.filters, marr.T))

    # FIXME: could apply distance modulus to all mags and pass apparent mag
    # arrays to `masker`
    mag = mags[band] + mu
    if masker is not None:
        mask = masker(params, mags)
    else:
        mask = True
    valid = (mag > bright) & (mag < faint) & mask
    if wisecut:
        w1 = mags["WISE_W1"] + mu
        valid = valid & (w1 < 15.5)
    frac = np.nansum(weights[valid])
    return frac


def logg_masker(params, mags):
    """Default: identify giants based on logg, remove  K Giants.
    """
    sel = params["logg"] < 3.5
    sel = sel & (~mgiant_masker(params, mags))
    return sel


def mgiant_masker(params, mags):
    """Identify M(K) Giants basd on Conroy+ 2019.  Does not include apparent mag
    cut in WISE_W1, but does include logg cut
    """
    g_r = mags["PS_g"] - mags["PS_r"]
    w1_w2 = mags["WISE_W1"] - mags["WISE_W2"]
    z_w1 = mags["PS_z"] - mags["WISE_W1"]
    sel = ((w1_w2 > -0.4) & (w1_w2 < 0.) &
           (g_r < 1.1) &
           (z_w1 > 1.9) & (z_w1 < 2.5) &
           (z_w1 > (1.2*g_r + 0.95)) & (z_w1 < (1.2*g_r + 1.15)) &
           (params["logg"] < 3.5))
    return sel


def bhb_masker(params, mags):
    """Identify BHB stars based on EEPs
    """
    # Deason 2014 color cuts
    #ps_g_r = mags["PS_g"] - mags["PS_r"]
    #g_r = mags["SDSS_g"] - mags["SDSS_r"]
    #u_g = mags["SDSS_u"] - mags["SDSS_g"]
    #ugbhb = 1.167 - 0.775*g_r - 1.934*g_r**2 + 9.936*g_r**3
    #sel = ((g_r > -2.5) & (g_r < 0.) &
    #       (np.abs(u_g - ugbhb) < 0.08) & 
    #       (ps_g_r < 0.05))

    # Just cut on EEP
    sel = (params["eep"] < 707) & (params["eep"] > 631)

    return sel


def rank_weight(rank, ptgID, wcat=None):
    """Compute the fraction of stars that made it from mag selected sample
    (scat) into rcat given rank and ptgID
    """
    ind = wcat["PTGID"] == ptgID
    # probability to go from scat to rcat for this rank
    w1 = wcat[ind]["RANK{:.0f}_WGT_ALL".format(rank)]
    return w1


def compute_total_weight(rcat_row, iso, wcat, snr_limit=3.0, use_ebv=False, 
                         lf_params=dict(afe=0, loga=10)):
    """Compute the fiber assignment weight and the magnitude weighting for this star.
    The inverse of the product of these weights represents the number of stars
    that would be required to produce a single expected star at this distance
    and FeH in the rcat.
    
    Parameters
    ----------
    rcat_row : length 1 FITS_rec or structured array
        A row in the rcat, must be a giant (logg < 3.5) and have SNR < snr_limit
    
    iso : brutus.seds.Isochrone instance
        must have the PS and WISE filters

    wcat : FITS_rec or structured array
        pcat with weights
    
    snr_limit : float (optional, default=3.)
        The limiting SNR used to select this row.

    use_ebv : bool (optional, default=False)
        Adjust the weights to account for a foreground screen (EBV) when
        computing the observability of isochrone points.
    
    lf_params : dict (optional)
        Parameters (e.g. `loga`, `afe` used to construct the isochrone; weights
        will be appropriate for a population described by these parameters.
        
    Returns
    -------
    rank_weight : float
        Number between 0 and 1 that gives the probability a star of this rank
        had a fiber assigned.

    mag_weight :  float
        Number that gives the fraction of stars in an SSP with the
        same distance and FeH of the star that would have been selected for the
        scat and observed above the given SNR threshold.
    """
    assert rcat_row["logg"] < 3.5
    assert rcat_row["SNR"] >= snr_limit
    row = rcat_row
    
    # probability to go from scat to rcat for this rank
    ptgID = "{}_{}_{}".format(row["tileID"], row["selID"], row["dateID"])
    ptg_ind = wcat["PTGID"] == ptgID
    rank = row["XFIT_RANK"]
    rw = wcat[ptg_ind]["RANK{:.0f}_WGT_ALL".format(rank)]
    
    # find PS_r s.t. SNR=3 in this ptg
    coeffs = wcat[ptg_ind]["snr_curve"][0].copy()
    coeffs[-1] -= snr_limit
    faint_snr = min(np.roots(coeffs)).real

    masker = logg_masker
    
    # get bright and faint limits for this rank, ptg
    band, wisecut = "PS_r", False
    if rank == 3:
        faint = min(18.5, faint_snr)
        bright = 18
    elif (rank == 1):
        faint = min(17.5, faint_snr)
        bright = 13.5
    else:
        faint = min(18.0, faint_snr)
        bright = 15.0
    
    # Account for reddening in the limits
    if use_ebv:
        a_r = row["EBV"] * 2.271
        bright -= a_r
        faint -= a_r
    
    # get isochrone masker for this selection
    if (rank == 1) & (row["MGIANT"] > 0):
        masker = mgiant_masker
        wisecut = True
        #band = "WISE_W1"
    elif (rank == 1) & (row["BHB"] > 0):
        masker = bhb_masker

    mw = mag_weight(row["dist_adpt"], row["FeH"], iso,
                    bright=bright, faint=faint, selection_band=band,
                    wisecut=wisecut, masker=masker,
                    lf_params=lf_params)

    return rw, mw, faint_snr


if __name__ == "__main__":

    from archer.seds import Isochrone, FILTERS
    from archer.config import parser, rectify_config, plot_defaults
    _ = plot_defaults(pl.rcParams)
    config = rectify_config(parser.parse_args())

    # test the mag weight code
    feh = -1.0
    filters = np.array(["PS_r", "PS_g", "PS_z", 
                        "SDSS_u", "SDSS_g", "SDSS_r",
                        "WISE_W1", "WISE_W2"])
    iso = Isochrone(mistfile=config.mistiso, nnfile=config.nnfile, filters=filters)
    mags, params, weights = get_lf(iso, feh=feh)
    gmask = logg_masker(params, mags)
    mmask = mgiant_masker(params, mags)
    bmask = bhb_masker(params, mags)

    d_arr = np.linspace(2, 100, 1000)
    frac_g = lf_observable_fraction(mags, weights, d_arr, mask=gmask)
    frac_r3 = lf_observable_fraction(mags, weights, d_arr, mask=gmask,
                                     faint=18.5, bright=18.0,)
    frac_m = lf_observable_fraction(mags, weights, d_arr, mask=mmask,
                                    faint=17.5, bright=13.5, wisecut=True)
    frac_b = lf_observable_fraction(mags, weights, d_arr, mask=bmask,
                                    faint=17.5, bright=13.5)
    frac = lf_observable_fraction(mags, weights, d_arr)


    fig, ax = pl.subplots()
    ax.plot(d_arr, np.log10(frac) - np.log10(frac[0]), label="All stars")
    ax.plot(d_arr, np.log10(frac_g) - np.log10(frac[0]), label="logg < 3.5")
    ax.plot(d_arr, np.log10(frac_m) - np.log10(frac[0]), label="M Giants")
    ax.plot(d_arr, np.log10(frac_b) - np.log10(frac[0]), label="BHB")
    ax.plot(d_arr, np.log10(frac_r3) - np.log10(frac[0]), label="rank=3 logg < 3.5")
    ax.set_xscale("log")
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel(r"log (N$_{\rm targetable}$ / N) + const.")
    ax.legend()
    ax.set_xlim(2, 50)
    ax.set_ylim(-5, 0.2)
    ax.set_title("[Fe/H] = {:3.1f}".format(feh))
    pl.show()
    fig.savefig("mag_weighting_feh{:+3.1f}.png".format(feh), dpi=450)  

    import sys
    sys.exit()

    from astropy.io import fits
    rcat = fits.getdata(config.rcat_file)
    pcat = None
    for row in rcat[good & sgr]:
        if row["BHB"] == 0:
            rw, mw = compute_total_weight(row, iso, pcat, snr_limit=3)
