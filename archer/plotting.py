#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import Normalize


def make_cuts(ax, delt=0.015, angle=1.0, right=True):
    """Put diagonal lines on the ends of a set of axes to indicate a break

    Parameters
    ----------
    delt: float
        how big to make the diagonal lines in axes coordinates

    angle : float, optional (default: 1.0)
        increase to get steeper lines
    """

    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    d = np.array([-delt, delt])

    if right:
        ax.plot(1 + d/angle, d, **kwargs)
        ax.plot(1 + d/angle, 1+d, **kwargs)
    else:
        ax.plot(d/angle, 1+d, **kwargs)
        ax.plot(d/angle, d, **kwargs)
    return ax


def hquiver(cat_r, sel, colorby=None, nshow=None, randomize=True,
            axes="yz", ax=None, scale=20, vmin=None, vmax=None, **quiver_kwargs):

    if randomize:
        if nshow is None:
            nshow = sel.sum()
        rand = np.random.choice(sel.sum(), size=nshow, replace=False)
    else:
        rand = slice(None)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- Get quantities ---
    x, y = [cat_r["{}_gal".format(s)] for s in axes]
    vx, vy = [cat_r["v{}_gal".format(s)] for s in axes]
    vsq = np.array([cat_r["v{}_gal".format(s)]**2 for s in "xyz"])
    vtot = np.sqrt(vsq.sum(axis=0))

    # --- plot them ---
    if colorby is not None:
        cb = ax.quiver(x[sel][rand], y[sel][rand],
                       (vx/vtot)[sel][rand], (vy/vtot)[sel][rand],
                       colorby[sel][rand], norm=norm,
                       angles="xy", pivot="mid",
                       scale_units="height", scale=scale,
                       **quiver_kwargs)
    else:
        cb = ax.quiver(x[sel][rand], y[sel][rand],
                       (vx/vtot)[sel][rand], (vy/vtot)[sel][rand],
                       angles="xy", pivot="mid",
                       vmin=vmin, vmax=vmax,
                       scale_units="height", scale=scale,
                       **quiver_kwargs)

    return ax, cb


def hnoquiver(cat_r, sel, colorby=None, nshow=None, randomize=True,
              axes="yz", ax=None, scale=20, **plot_kwargs):

    if randomize:
        if nshow is None:
            nshow = sel.sum()
        rand = np.random.choice(sel.sum(), size=nshow, replace=False)
    else:
        rand = slice(None)

    # --- Get quantities ---
    x, y = [cat_r["{}_gal".format(s)] for s in axes]

    # --- plot them ---
    if colorby is not None:
        cb = ax.scatter(x[sel][rand], y[sel][rand], c=colorby[sel][rand],
                        s=scale, **plot_kwargs)
    else:
        cb = ax.plot(x[sel][rand], y[sel][rand],
                     s=scale, **plot_kwargs)

    return ax, cb
