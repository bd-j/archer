#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from .seds import Isochrone, FILTERS


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


def observable_fraction(mag, wght, d_kpc, mask=True, faint=18, bright=15):
    """Given an LF ([mag, weight] pair), compute corrections for a range of distances
    """
    mu = 5 * np.log10(d_kpc) + 10
    frac = np.zeros_like(mu)
    for i, m in enumerate(mu):
        mags = mag + m
        valid = (mags > bright) & (mags < faint) & mask
        frac[i] = np.nansum(wght[valid])
    return frac


def correction(dist, feh, iso, masker=None,
               bright=15, faint=18,
               lf_params=dict(afe=0, loga=10)):
    """Compute correction for a single (distance, feh) pair
    """
    mu = 5 * np.log10(dist) + 10
    mags, params, _ = iso.get_seds(feh=feh, apply_corr=False, dist=10,
                                   **lf_params)
    weights = imf_weight(params["mini"])
    mag = mags[:, 0] + mu
    if masker is not None:
        mask = masker(params, mags)
    else:
        mask = True
    valid = (mag > bright) & (mag < faint) & mask
    frac = np.nansum(weights[valid])
    return frac


def correction_table(iso, distances, metallicities, masker=None,
                     lf_params=dict(afe=0, loga=10),
                     obs_params=dict(bright=15, faint=18)):
    frac = np.zeros([len(metallicities), len(distances)])
    for i, feh in enumerate(metallicities):
        mags, params, weights = get_lf(iso, feh=feh, **lf_params)
        if masker is not None:
            mask = masker(params)
        else:
            mask = True
        frac[i, :] = observable_fraction(mags[:, 0], weights, distances,
                                         mask=mask, **obs_params)
    return frac


def gmasker(params, mags):
    """Default apply mask based on logg
    """
    return params["logg"] < 3.5


if __name__ == "__main__":
    from archer.config import parser, rectify_config
    config = rectify_config(parser.parse_args())
    
    filters = np.array(["PS_r"])
    iso = Isochrone(mistfile=config.mistiso, nnfile=config.nnfile, filters=filters)
    mags, params, weights = get_lf(iso, feh=-1)
    gmask = (params["logg"] < 3.5)

    d_arr = np.linspace(2, 100, 1000)
    frac_g = observable_fraction(mags[:, 0], weights, d_arr, mask=gmask)
    frac = observable_fraction(mags[:, 0], weights, d_arr)

    fig, ax = pl.subplots()
    ax.plot(d_arr, np.log10(frac) - np.log10(frac[0]))
    ax.plot(d_arr, np.log10(frac_g) - np.log10(frac[0]))
    ax.set_xscale("log")
    pl.show()
