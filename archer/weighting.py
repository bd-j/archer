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
    wght /= np.nansum(wght)
    return wght


def get_lf(iso, loga=10, feh=-1, afe=0):
    mags, params, _ = iso.get_seds(loga=loga, feh=feh, afe=afe,
                                   apply_corr=False, dist=10)
    weights = imf_weight(params["mini"])
    return mags, params, weights


def lf_observable_fraction(mag, wght, d_kpc, mask=True, faint=18, bright=15):
    """Given an LF ([mag, weight] pair), compute corrections for a range of distances
    """
    mu = 5 * np.log10(d_kpc) + 10
    frac = np.zeros_like(mu)
    for i, m in enumerate(mu):
        mags = mag + m
        valid = (mags > bright) & (mags < faint) & mask
        frac[i] = np.nansum(wght[valid])
    return frac


def mag_weight(dist, feh, iso, masker=None,
               bright=15, faint=18, selection_band="PS_r",
               lf_params=dict(afe=0, loga=10)):
    """Compute probability that a star of a given feh and distance made it
    into the magnitude selection, based on the LF for that metallicity.
    Takes about 50ms per object

    dist : float
        Distance in kpc

    feh : float
        [Fe/H] of the target

    iso : seds.Isochrone instance
        Assumes that the mag selection band (i.e. PS_r) is the first element
        of the output magnitudes

    masker : callable, optional
        A function that takes a params structured array and a set of apparent
        magnitudes and returns a boolean array indicating whether each element
        (isochrone point) is a valid member

    bright : float (optional, default 15)
        Bright limit to have made it into the selection.

    faint : float (optional, default 18)
        Faint limit to have made it into the selection

    lf_params : dict
        Dictionary of parameters (e.g. loga, afe) for generating the isochrone & LF
    """
    band = iso.filters == selection_band
    assert band.sum() == 1

    mu = 5 * np.log10(dist) + 10
    mags, params, _ = iso.get_seds(feh=feh, apply_corr=False, dist=10,
                                   **lf_params)
    weights = imf_weight(params["mini"])
    mag = mags[:, band] + mu
    if masker is not None:
        mask = masker(params, mags, iso.filters.tolist())
    else:
        mask = True
    valid = (mag > bright) & (mag < faint) & mask
    frac = np.nansum(weights[valid])
    return frac


def logg_masker(params, mags, filters):
    """Default: apply mask based on logg
    """
    return params["logg"] < 3.5


def mgiant_masker(params, mags, filters):
    
    g_r = mags[:, filters.index("PS_g")] - mags[:, filters.index("PS_r")]
    w1_w2 = mags[:, filters.index("WISE_W1")] - mags[:, filters.index("WISE_W2")]
    z_w1 = mags[:, filters.index("PS_z")] - mags[:, filters.index("WISE_W1")]
    sel = ((w1_w2 > -0.4) & (w1_w2 < 0.) &
           (g_r < 1.1) &
           (z_w1 > 1.9) & (z_w1 < 2.5) &
           (z_w1 > (1.2*g_r + 0.95)) & (z_w1 < (1.2*g_r + 1.15))
           )
    return sel


def bhb_masker(params, mags, filters):
    return None


def rank_weight(rank, ptgID, wcat=None, selector=None, rcat=None):
    """Compute the fraction of stars that made it from mag selected sample
    (scat) into parent sample of interest given rank and ptgID

    wcat : FITS_rec or structured array
        pcat with weights

    selector : optional, callable
        A function that applies the final 'parent' sample selection
        (e.g. logg < 3.5, snr > 3).  If not supplied, final weights will come
        from the wcat
    
    rcat : optional FITS_rec or structured array
        the full rcat.  if not supplied, final weights will come from the wcat
    """
    ind = wcat["PTGID"] == ptgID
    # probability to go from scat to rcat for this rank
    w1 = wcat[ind]["RANK{}_WGT_ALL".format(rank)]
    
    # probability to go from rcat to selection for this rank
    #if (rcat is None) or (selector is None):
    #    w1 = wcat[ind]["RANK{}_WGT_ALL".format(rank)]
    #    w2 = 1.0
    #else:
    #    # compute from rcat
    #    tileID, selID, dateID = ptgID.split('_')
    #    inds = ((rcat["tileID"] == tileID) & (rcat["selID"] == selID) &
    #            (rcat["dateID"] == dateID) & (rcat["XFIT_RANK"] == rank))
    #    # number in rcat with this rank
    #    n_ptg = inds.sum()
    #    # number selected from rcat with this rank
    #   n_sel = selector(rcat[inds]).sum()
    #   w2 = n_sel*1. / n_ptg
    return w1


def compute_total_weight(rcat_row, iso, wcat, snr_limit=3.0):
    row = rcat_row
    ptgID = "{}_{}_{}".format(row["tileID"], row["selID"], row["dateID"])
    ptg_ind = wcat["PTGID"] == ptgID
    rank = row["XFIT_RANK"]
    rw = wcat[ptg_ind]["RANK{}_WGT_ALL".format(rank)]
    
    # find PS_r s.t. SNR=3
    coeffs = wcat[ptg_ind]["snr_curve"][0].copy()
    coeffs[-1] -= snr_limit
    faint_snr = min(np.roots(coeffs))

    masker = logg_masker
    
    # get bright and faint limits for this rank
    band = "PS_r"
    if rank == 3:
        faint = min(18.5, faint_snr)
        bright = 18
    elif (rank == 1):
        faint = min(17.5, faint_snr)
        bright = 13.5
    else:
        faint = min(18.0, faint_snr)
        bright = 15.0
    
    # get isochrone masker for this selection
    if (rank == 1) & (row["MGIANT"] > 0):
        masker = mgiant_masker
        #band = "WISE_W1"
    elif (rank == 1) & (row["BHB"] > 0):
        masker = bhb_masker

    mw = mag_weight(row["dist_adpt"], row["FeH"], iso,
                    bright=bright, faint=faint, selection_band=band,
                    masker=masker)

    return rw, mw


if __name__ == "__main__":

    from archer.seds import Isochrone, FILTERS
    from archer.config import parser, rectify_config
    config = rectify_config(parser.parse_args())

    # test the mag weight code
    feh = -2.0
    filters = np.array(["PS_r", "PS_g", "PS_z", "WISE_W1", "WISE_W2"])
    iso = Isochrone(mistfile=config.mistiso, nnfile=config.nnfile, filters=filters)
    mags, params, weights = get_lf(iso, feh=feh)
    gmask = logg_masker(params, mags, iso.filters.tolist())
    mmask = mgiant_masker(params, mags, iso.filters.tolist())

    d_arr = np.linspace(2, 100, 1000)
    frac_g = lf_observable_fraction(mags[:, 0], weights, d_arr, mask=gmask)
    frac_m = lf_observable_fraction(mags[:, 0], weights, d_arr, mask=mmask,
                                    faint=17.5, bright=13.5)
    frac = lf_observable_fraction(mags[:, 0], weights, d_arr)


    fig, ax = pl.subplots()
    ax.plot(d_arr, np.log10(frac) - np.log10(frac[0]), label="All stars")
    ax.plot(d_arr, np.log10(frac_g) - np.log10(frac[0]), label="logg < 3.5")
    ax.plot(d_arr, np.log10(frac_m) - np.log10(frac[0]), label="M Giants")
    ax.set_xscale("log")
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel(r"log (N$_{\rm targetable}$ / N)")
    ax.legend()
    ax.set_xlim(2, 50)
    ax.set_ylim(-5, 0.2)
    ax.set_title("[Fe/H] = {:3.1f}".format(feh))
    pl.show()

    import sys
    sys.exit()

    from astropy.io import fits
    rcat = fits.getdata(config.rcat_file)
    pcat = None
    for row in rcat[good & sgr]:
        if row["BHB"] == 0:
            rw, mw = compute_total_weight(row, iso, pcat, snr_limit=3)
