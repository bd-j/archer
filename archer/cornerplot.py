#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle

from scipy.ndimage import gaussian_filter as norm_kde


__all__ = ["_quantile", "quantile", "get_spans", "get_cmap", "twodhist"]


def get_spans(span, samples, weights=None):
    """Get ranges from percentiles of samples
    """
    ndim = len(samples)
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != len(samples):
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except(TypeError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)
    return span


def quantile(xarr, q, weights=None):
   qq = [_quantile(x, q, weights=weights) for x in xarr]
   return np.array(qq)


def _quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles


def _hist2d(*args, **kwargs):
    """backwards compat"""
    return twodhist(*args, **kwargs)

def twodhist(x, y, ax=None, span=None, weights=None,
             smooth=0.02, levels=None, color='gray',
             plot_density=False, plot_contours=True, fill_contours=True,
             contour_kwargs={}, contourf_kwargs={}, **kwargs):

    # Determine plotting bounds.
    span = get_spans(span, [x, y], weights=weights)
    # Setting up smoothing.
    smooth = np.zeros(2) + smooth

    # --- Now actually do the plotting-------

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)
    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    contour_cmap = get_cmap(color, levels)
    # Initialize smoothing.
    smooth = np.zeros(2) + np.array(smooth)
    bins = []
    svalues = []
    for s in smooth:
        if s > 1.0:
            # If `s` > 1.0, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(int(s))
            svalues.append(0.)
        else:
            # If `s` < 1, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results.
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        print("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    clevels = np.concatenate([[0], V, [H.max() * (1 + 1e-4)]])
    # plot contour fills
    if plot_contours and fill_contours and (ax is not None):
        cfk = {}
        cfk["colors"] = contour_cmap
        cfk["antialiased"] = False
        cfk.update(contourf_kwargs)
        ax.contourf(X2, Y2, H2.T, clevels, **cfk)

    # Plot the contour edge colors.
    if plot_contours and (ax is not None):
        ck = {}
        ck["colors"] = color
        ck.update(contour_kwargs)
        ax.contour(X2, Y2, H2.T, V, **ck)

    return X2, Y2, H2.T, V, clevels, ax


def get_cmap(color, levels):
    nl = len(levels)
    from matplotlib.colors import colorConverter
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [list(rgba_color)]
    for i in range(nl+1):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)
    return contour_cmap


def marginal(x, ax=None, weights=None, span=None, smooth=0.02,
             color='black', peak=None, **hist_kwargs):

    if span is None:
        span = get_spans(span, np.atleast_2d(x), weights=weights)[0]
    ax.set_xlim(span)

    # Generate distribution.
    if smooth > 1:
        # If `sx` > 1, plot a weighted histogram
        #n, b, _ = ax.hist(x, bins=smooth, weights=weights, range=np.sort(span),
        #                  color=color, **hist_kwargs)
        #n, b = np.histogram(x, bins=smooth, weights=weights, range=np.sort(span))
        xx, bins, wght = x, int(round(smooth)), weights
    else:
        # If `sx` < 1, oversample the data relative to the
        # smoothing filter by a factor of 10, then use a Gaussian
        # filter to smooth the results.
        bins = int(round(10. / smooth))
        n, b = np.histogram(x, bins=bins, weights=weights,
                            range=np.sort(span))
        n = norm_kde(n, 10.)
        b0 = 0.5 * (b[1:] + b[:-1])
        #n, b, _ = ax.hist(b0, bins=b, weights=n, range=np.sort(span),
        #                  color=color, **hist_kwargs)
        #n, b = np.histogram(b0, bins=b, weights=n, range=np.sort(span))
        xx, bins, wght = b0, b, n

    n, b = np.histogram(xx, bins=bins, weights=wght, range=np.sort(span))
    if peak is not None:
        wght = wght * peak / n.max()

    n, b, _ = ax.hist(xx, bins=bins, weights=wght, range=np.sort(span),
                      color=color, **hist_kwargs)
    ax.set_ylim([0., max(n) * 1.05])
