#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""weighting.py - logic for computing weights of stars in the rcat
"""

import numpy as np
import matplotlib.pyplot as pl
from functools import partial

__all__ = ["eta_iso", "show_weights",
           "compute_total_weight", "mag_weight", "rank_weight",
           "logg_masker", "mgiant_masker", "bhb_masker",
           "kroupa_imf", "imf_weight",
           "EBV"]


EBV = dict([("PS_{}".format(b), e)
            for b, e in zip("grizy", [3.172, 2.271, 1.682, 1.322, 1.087])])
EBV.update(dict([("SDSS_{}".format(b), e)
           for b, e in zip("ugriz", [4.239, 3.303, 2.285, 1.698, 1.263])]))
EBV.update(dict([("TMASS_{}".format(b), e)
           for b, e in zip("JHK", [0.65, 0.33, 0.2])]))
EBV["GAIA_G"] = 2.79
EBV["WISE_W1"] = 0.18
EBV["WISE_W2"] = 0.16


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


def mag_weight(dist, feh, iso, masker=None,
               bright=15, faint=18, selection_band="PS_r",
               lf_params=dict(afe=0, loga=10)):
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
    mags = dict(zip(iso.filters, marr.T + mu))

    mag = mags[band]
    if masker is not None:
        mask = masker(params, mags)
    else:
        mask = True
    valid = (mag > bright) & (mag < faint) & mask
    frac = np.nansum(weights[valid])
    return frac


def logg_masker(params, mags, faint={}, w1cut=True):
    """Default: identify giants based on logg, remove  K Giants.
    """
    sel = params["logg"] < 3.5
    sel = sel & (~mgiant_masker(params, mags, w1cut=w1cut))
    if len(faint) > 0:
        band, limit = list(faint.items())[0]
        sel = sel & (mags[band] < limit)
    return sel


def mgiant_masker(params, mags, w1cut=True, faint={}):
    """Identify M(K) Giants basd on Conroy+ 2019.  Includes logg cut
    """
    g_r = mags["PS_g"] - mags["PS_r"]
    w1_w2 = mags["WISE_W1"] - mags["WISE_W2"]
    z_w1 = mags["PS_z"] - mags["WISE_W1"]
    with np.errstate(invalid="ignore"):
        sel = ((w1_w2 > -0.4) & (w1_w2 < 0.) &
               (g_r < 1.1) &
               (z_w1 > 1.9) & (z_w1 < 2.5) &
               (z_w1 > (1.2*g_r + 0.95)) & (z_w1 < (1.2*g_r + 1.15)) &
               (params["logg"] < 3.5))
        if w1cut:
            sel = sel & (mags["WISE_W1"] < 15.5)
        if len(faint) > 0:
            band, limit = list(faint.items())[0]
            sel = sel & (mags[band] < limit)
    return sel


def bhb_masker(params, mags, faint={}):
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
    with np.errstate(invalid="ignore"):
        sel = (params["eep"] < 707) & (params["eep"] > 631)
        if len(faint) > 0:
            band, limit = list(faint.items())[0]
            sel = sel & (mags[band] < limit)

    return sel


def rank_weight(rank, ptgID, pcat=None):
    """Compute the fraction of stars that made it from mag selected sample
    (scat) into rcat given rank and ptgID
    """
    ind = pcat["PTGID"] == ptgID
    # probability to go from scat to rcat for this rank
    w1 = pcat[ind]["RANK{:.0f}_WGT_ALL".format(rank)]
    return w1


def ptg_limit(ptgID, pcat, snr_limit, rcat=None, band="g"):
    if rcat is not None:
        tileID, selID, dateID = ptgID.split("_")
        inptg = ((rcat["tileID"] == tileID) &
                 (rcat["selID"] == selID) &
                 (rcat["dateID"] == dateID) &
                 (rcat["SNR"] > snr_limit) &
                 (rcat["Ps_{}".format(band)] < 80)
                )
        faint = np.max(rcat[inptg]["PS_{}".format(band)])
    else:
        ptg_ind = pcat["PTGID"] == ptgID
        if band == "g":
            col = "snr_curve_g"
        else:
            col = "snr_curve"
        coeffs = pcat[ptg_ind][col][0].copy()
        coeffs[-1] -= snr_limit
        faint = min(np.roots(coeffs)).real

    return faint


def compute_total_weight(rcat_row, iso, pcat, snr_limit=3.0, limit_band="g", rcat=None,
                         use_ebv=False, use_afe=False, use_age=False,
                         lf_params=dict(afe=0, loga=10), delta_m=0):
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

    pcat : FITS_rec or structured array
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
    eta_rank : float
        Number between 0 and 1 that gives the probability a star of this rank
        had a fiber assigned.

    eta_mag :  float
        Number that gives the fraction of stars in an SSP with the
        same distance and FeH of the star that would have been selected for the
        scat and observed above the given SNR threshold.

    faintmag : float
        The limiting magnitude corresponding to the SNR limit
    """
    assert rcat_row["logg"] < 3.5
    assert rcat_row["SNR"] >= snr_limit
    row = rcat_row

    # probability to go from scat to rcat for this rank
    ptgID = "{}_{}_{}".format(row["tileID"], row["selID"], row["dateID"])
    ptg_ind = pcat["PTGID"] == ptgID
    rank = row["XFIT_RANK"]
    er = pcat[ptg_ind]["RANK{:.0f}_WGT_ALL".format(rank)]

    # find PS_g s.t. SNR=snr_limit in this ptg
    lim_band = "PS_{}".format(limit_band)
    faintmag = ptg_limit(ptgID, pcat, snr_limit, rcat=rcat, band=limit_band)
    # make sure the star is brighter than the faint limit
    faintmag = max(faintmag, row[lim_band])
    # add a fudge to the limit?
    faintmag += delta_m

    # get bright and faint limits for this rank, ptg
    band, bright, faint = "PS_r", 15.0, 18.0
    if rank == 3:
        faint = 18.5
        bright = 18
    elif (rank == 1):
        faint = 17.5
        bright = 13.5

    # De-reddening the limits
    if use_ebv:
        Ab = row["EBV"] * EBV[band]
        bright -= Ab
        faint -= Ab
        faintmag -= row["EBV"] * EBV[lim_band]

    # get isochrone masker for this selection
    faintlim = {lim_band: faintmag}
    masker = partial(logg_masker, faint=faintlim)
    if (rank == 1) & (row["MGIANT"] > 0):
        masker = partial(mgiant_masker, faint=faintlim)
    elif (rank == 1) & (row["BHB"] > 0):
        masker = partial(bhb_masker, faint=faintlim)

    # choose afe and age
    if use_afe:
        lf_params["afe"] = row["init_aFe"]
    if use_age:
        lf_params["loga"] = row["logAge"]

    em = mag_weight(row["dist_adpt"], row["init_FeH"], iso,
                    bright=bright, faint=faint, selection_band=band,
                    masker=masker, lf_params=lf_params)

    return er, em, faintmag


def eta_iso(distances, feh, iso, masker=None,
            bright=15, faint=18, band="PS_r",
            lf_params=dict(afe=0, loga=10),):
    """Construct an eta_mag vs distance curve
    """
    eta = np.zeros_like(distances)
    for i, dist in enumerate(distances):
        eta[i] = mag_weight(dist, feh, iso, masker=masker,
                            bright=bright, faint=faint, selection_band=band,
                            lf_params=lf_params)
    return eta


def show_weights(d_arr=np.linspace(2, 50, 100), feh=-1, iso=None):
    """Make a plot showing the eta_mag v distance curves for different ranks
    """
    frac_all    = eta_iso(d_arr, feh, iso)
    frac_giants = eta_iso(d_arr, feh, iso, masker=logg_masker)
    frac_rank3  = eta_iso(d_arr, feh, iso, masker=logg_masker,
                          faint=18.5, bright=18.0,)
    frac_mgiant = eta_iso(d_arr, feh, iso, masker=mgiant_masker,
                          faint=17.5, bright=13.5)
    frac_bhbs   = eta_iso(d_arr, feh, iso, masker=bhb_masker,
                          faint=17.5, bright=13.5)

    fig, ax = pl.subplots()
    logf0 = np.log10(frac_all[0])
    ax.plot(d_arr, np.log10(frac_all) - logf0, label="All stars")
    ax.plot(d_arr, np.log10(frac_giants) - logf0, label="logg < 3.5")
    ax.plot(d_arr, np.log10(frac_mgiant) - logf0, label="M Giants")
    ax.plot(d_arr, np.log10(frac_bhbs) - logf0, label="BHB")
    ax.plot(d_arr, np.log10(frac_rank3) - logf0, label="rank=3 logg < 3.5")
    ax.set_xscale("log")
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel(r"log (N$_{\rm targetable}$ / N) + const.")
    ax.legend()
    ax.set_xlim(2, 50)
    ax.set_ylim(-5, 0.2)
    ax.set_title("[Fe/H] = {:3.1f}".format(feh))
    return fig, ax


if __name__ == "__main__":

    import os, time
    from archer.seds import Isochrone, FILTERS
    from archer.config import parser, rectify_config, plot_defaults
    _ = plot_defaults(pl.rcParams)

    try:
        parser.add_argument("--test", action="store_true")
        parser.add_argument("--snr_limit", type=float, default=3.0)
        parser.add_argument("--limit_band", type=str, default="r",
                            help=("band for computing limiting snr, 'r' | 'g'"))
        parser.add_argument("--use_ebv", action="store_true")
        parser.add_argument("--use_afe", action="store_true")
        parser.add_argument("--use_age", action="store_true")
    except:
        pass

    config = rectify_config(parser.parse_args())

    # test the mag weight code
    if config.test:
        filters = np.array(["PS_r", "PS_g", "PS_z",
                            "SDSS_u", "SDSS_g", "SDSS_r",
                            "WISE_W1", "WISE_W2"])
        iso = Isochrone(mistfile=config.mistiso, nnfile=config.nnfile,
                        filters=filters)
        fig, ax = show_weights(feh=-1.0, iso=iso)
        pl.ion()
        pl.show()

    from astropy.io import fits
    rcat = fits.getdata(config.rcat_file)
    pcat = fits.getdata(config.pcat_file)

    # output wcat name
    outname = config.rcat_file.replace("rcat", "wcat")
    options = {"ebv": config.use_ebv, "alpha": config.use_alpha, "age": config.use_age}
    tag = "_".join([k for k in options.keys() if options[k]])
    outname = outname.replace(".fits", "[{}_PS{}].fits".format(tag, config.limit_band))

    # select the stars to calculate weights for
    good = ((rcat["logg"] < 3.5) & (rcat["SNR"] > config.snr_limit) &
            ((rcat["FLAG"] == 0) | (rcat["BHB"] > 0)))

    # build the empty wcat
    lim_col = "{}mag_snr_limit".format(config.limit_band)
    colnames = ["rank_weight", "mag_weight", "total_weight", lim_col, "ebv"]
    cols = [rcat.dtype.descr[rcat.dtype.names.index("starname")]]
    cols += [(c, np.float) for c in colnames]
    dtype = np.dtype(cols)
    wcat = np.zeros(len(rcat), dtype=dtype)
    wcat["starname"] = rcat["starname"]
    wcat["ebv"][good] = rcat["EBV"][good]
    wcat["total_weight"][~good] = -1

    # compute the weights and add to wcat
    gg = np.where(good)[0]
    N = len(gg)
    for i, g in enumerate(gg):
        if np.mod(i, 1000) == 0:
            print(f"{i} of {N}")

        rw, mw, lim = compute_total_weight(rcat[g], iso, pcat, snr_limit=config.snr_limit,
                                           limit_band=config.limit_band, use_ebv=config.use_ebv,
                                           use_afe=config.use_afe, use_age=config.use_afe)
        wcat[g]["rank_weight"] = rw
        wcat[g]["mag_weight"] = mw
        wcat[g]["total_weight"] = mw * rw
        wcat[g][lim_col] = lim

    hdu = fits.BinTableHDU(wcat)
    hdu.header["SNRLIM"] = config.snr_limit
    hdu.header["LIMBAND"] = config.limit_band
    hdu.header["USE_EBV"] = options["ebv"]
    hdu.header["USE_AFE"] = options["alpha"]
    hdu.header["USE_AGE"] = options["age"]
    hdu.header["ISOC"] = os.path.basename(config.mistiso)
    hdu.header["SEDS"] = os.path.basename(config.nnfile)
    hdu.header["CREATED"] = time.asctime(time.localtime(time.time()))
    print(f"writing to {outname}")
    hdu.writeto(outname)
