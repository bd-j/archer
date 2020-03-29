#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

def phi(s):
    return -(1 + s**2)**(-1./2.)


def vesc(s):
    return np.sqrt(2) * (1 + s**2)**(-1/4.)


def rmax(e):
    return np.sqrt((-e)**(-2.) -1)


def energy(s, v):
    return v**2 / 2. + phi(s)


def radius(x):
    return (x**(-2./3.) - 1)**(-1./2.)


def g(x):
    return x**2 * (1-x**2)**(7./2.)


def rand(N):
    return np.random.uniform(0, 1, size=N)


def rank(e):
    estar = np.empty_like(e)
    estar[np.argsort(e)] = np.linspace(0, 1, len(estar))
    return estar


def make_particles(N=100000):
    """Following Aarseth (1974)
    """
    x4 = rand(N)
    x5 = rand(N)
    # rejection sample q from the distribution g
    good = 0.1 * x5 < g(x4)
    q = x4[good]

    x1 = rand(len(q))
    s = radius(x1)
    v = q * vesc(s)
    e = energy(s, v)
    
    return s, v, e


def convert_estar_rmax(estar):
    """Given the energy order of a particle (expressed as a float between
    0 and 1 with uniform distribution), determine the maximum radius for this particle in a plummer potential
    """
    s, v, e = make_particles()
    eorder = rank(e)
    o = np.argsort(eorder)
    eout = np.interp(estar, eorder[o], e[o])
    r = rmax(eout)
    return r, eout


if __name__ == "__main__":
    
    s, v, e = make_particles()
    rm = rmax(e)
    # the distribution of r/rmax is not a function of e
    rp = s/rm
    # it has mean and median at ~0.66
    print(np.median(rp), np.mean(rp))
    pct = np.percentile(rp, [16, 50, 84])


    estar = rank(e)
    o = np.argsort(estar)

    fig, axes = pl.subplots(1, 2)
    ax = axes[0]
    ax.plot(estar[o], e[o], marker="", color="k")
    ax.set_xlabel(r"E$_\ast$")
    ax.set_ylabel(r"E (normalized)")

    ax = axes[1]
    ax.plot(estar[o], rm[o] * pct[1], color="tomato", linewidth=2, marker="")
    ax.fill_between(estar[o], rm[o] * pct[0], rm[o]*pct[2], color="tomato", alpha=0.5)
    ax.set_ylim(1e-2, 20)
    ax.set_yscale("log")
    ax.set_xlabel(r"E$_\ast$")
    ax.set_ylabel(r"typical radius ($\sim 0.66 \, r_{\rm max}/r_0$)")
    fig.savefig("estar_energy_radius.png", dpi=450)  




