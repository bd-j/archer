#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl


def make_cuts(ax, delt=0.015, angle=1.0, right=True):
    # how big to make the diagonal lines in axes coordinates
    # angle = 1.0 # increase to get steeper lines
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

