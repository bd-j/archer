#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logsumexp


class VelocityModel:

    def __init__(self, alpha_range, beta_range, pout_range,
                 vmu_bad_range=None, vsig_bad_range=None):
        """
        Parameters
        ----------
        alpha_range : array_like, shape (2, alpha_order)
            min and max for alpha, 
            :math:`vmu = \sum_i \alpha_i \Lambda^i`

        beta_range : array_like, shape (2, neta_order)
            min and max for beta, 
            :math:`vsig = \sum_i \beta_i \Lambda^i`

        pout_range : sequence of length 2
            min and max for pout
        """
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.pout_range = np.array(pout_range)
        self.vmu_bad = -100
        self.vsig_bad = 200
        self.free_bad = False
        if vmu_bad_range is not None:
            self.free_bad = True
            self.vmu_bad_range = np.array(vmu_bad_range)
            self.vsig_bad_range = np.array(vsig_bad_range)
            assert vsig_bad_range is not None

    def pos_to_params(self, pos):
        """ {alpha}_i^N, {beta}_i^N, [vmu_bad, vsig_bad], pout
        """
        na, nb = self.alpha_order, self.beta_order
        self.alpha = pos[:na]
        self.beta = pos[na: (na + nb)]
        self.pout = pos[-1]
        if self.free_bad:
            self.vmu_bad = pos[-3]
            self.vsig_bad = pos[-2]

    def set_data(self, Lam, vel, vel_unc=0, idx=[-1]):
        self.lamb = Lam
        self.vel = vel
        self.vel_unc = vel_unc
        self.idx = idx

    def model(self, lam, pos=None):
        if pos is not None:
            self.pos_to_params(pos)
        return musig(lam, self.alpha, self.beta)

    def lnprob(self, pos):
        self.pos_to_params(pos)
        lnlike = model_lnlike(self.lamb, self.vel,
                              self.alpha, self.beta,
                              vel_unc=self.vel_unc, pout=self.pout,
                              vmu_bad=self.vmu_bad, vsig_bad=self.vsig_bad)
        assert np.isfinite(lnlike), "Infinite lnlike at {}".format(pos)
        return lnlike

    def prior_transform(self, u):
        na, nb = self.alpha_order, self.beta_order
        ar, br = self.alpha_range, self.beta_range
        a = ar[0] + u[:na] * np.diff(ar, axis=0)[0]
        b = br[0] + u[na:(na + nb)] * np.diff(br, axis=0)[0]
        pout = self.pout_range[0] + u[-1] * np.diff(self.pout_range)
        if self.free_bad:
            vm = self.vmu_bad_range[0] + u[-3] * np.diff(self.vmu_bad_range)
            vs = self.vsig_bad_range[0] + u[-2] * np.diff(self.vsig_bad_range)
            pars = np.hstack([a, b, vm, vs, pout])
        else:    
            pars = np.hstack([a, b, pout])
        return pars

    def outlier_odds(self, pos):
        """Compute the odds ratio of any given data point being an outlier for
        a particular model parameter position.
        """
        self.pos_to_params(pos)
        lnodds = outlier_oddsratio(self.lamb, self.vel,
                                   self.alpha, self.beta,
                                   vel_unc=self.vel_unc, pout=self.pout,
                                   vmu_bad=self.vmu_bad,
                                   vsig_bad=self.vsig_bad)
        return lnodds

    @property
    def alpha_order(self):
        return len(self.alpha_range[0])

    @property
    def beta_order(self):
        return len(self.beta_range[0])

    @property
    def ndim(self):
        return self.beta_order + self.alpha_order + 1 + 2 * self.free_bad


def musig(lam, alpha, beta):
    vmu = np.dot(alpha[::-1], np.vander(lam, len(alpha)).T)
    vsig = np.dot(beta[::-1], np.vander(lam, len(beta)).T)
    return vmu, vsig


def model_lnlike(lam, vel, alpha, beta, vel_unc=0, pout=0.0,
                 vmu_bad=-100, vsig_bad=200):

    vmu, vsig = musig(lam, alpha, beta)
    vvar = vsig**2 + vel_unc**2
    lnnorm = -np.log(np.sqrt(2 * np.pi * vvar))
    lnlike_good = lnnorm - 0.5 * ((vel - vmu)**2 / vvar)

    if pout > 0:
        vvar_bad = (np.zeros_like(vsig) + vsig_bad**2) + vel_unc**2
        bnorm = -np.log(np.sqrt(2 * np.pi * vvar_bad))
        lnlike_bad = bnorm - 0.5 * ((vel - vmu_bad)**2 / vvar_bad)
        a = np.log(1 - pout) + lnlike_good
        b = np.log(pout) + lnlike_bad
        lnlike = logsumexp(np.array([a, b]), axis=0)
        return lnlike.sum()
    else:
        return lnlike_good.sum()


def outlier_oddsratio(lam, vel, alpha, beta, vel_unc=0, pout=0.0,
                      vmu_bad=-100, vsig_bad=200):

    vmu, vsig = musig(lam, alpha, beta)
    vvar = vsig**2 + vel_unc**2
    lnnorm = -np.log(np.sqrt(2 * np.pi * vvar))
    lnlike_good = lnnorm - 0.5 * ((vel - vmu)**2 / vvar)

    vvar_bad = (np.zeros_like(vsig) + vsig_bad**2) + vel_unc**2
    bnorm = -np.log(np.sqrt(2 * np.pi * vvar_bad))
    lnlike_bad = bnorm - 0.5 * ((vel - vmu_bad)**2 / vvar_bad)

    lnodds = np.log(pout) + lnlike_bad - np.log(1 - pout) - lnlike_good

    return lnodds


def write_to_h5(results, model, oname):
    model_columns = ["alpha_range", "beta_range", "pout_range", 
                     "vmu_bad_range", "vsig_bad_range",
                     "lamb", "vel", "idx"]
    import h5py
    with h5py.File(oname, "w") as out:
        for mc in model_columns:
            d = getattr(model, mc)
            if d is not None:
                out.create_dataset(mc, data=d)
        for k, v in results.items():
            try:
                out.create_dataset(k, data=v)
            except:
                pass


def read_from_h5(iname):
    rcols = ["logl", "samples", "logz", "logwt"]
    import h5py
    with h5py.File(iname, "r") as f:
        if "vmu_bad_range" in f:
            vmb = f["vmu_bad_range"][:]
            vsb = f["vsig_bad_range"][:]
        else:
            vmb = vsb = None
        model = VelocityModel(f["alpha_range"][:], f["beta_range"][:],
                              f["pout_range"][:],
                              vmu_bad_range=vmb, vsig_bad_range=vsb)
        try:
            idx = f["idx"][:]
        except:
            idx = [-1]
        model.set_data(f["lamb"][:], f["vel"][:], idx=idx)
        results = {}
        for c in rcols:
            results[c] = f[c][:]

    return model, results


def best_model(filename, lam):
    model, r = read_from_h5(filename)
    pmax = r["samples"][r["logl"].argmax()]
    mu, sig = model.model(lam, pmax)
    return mu, np.abs(sig), pmax


def sample_posterior(filename, lam, n_sample=100):
    model, r = read_from_h5(filename)
    p = np.exp(r["logwt"] - r["logwt"].max())
    p /= p.sum()
    inds = np.random.choice(len(p), p=p, size=(n_sample,))
    pars = r["samples"][inds, :]

    mu = np.zeros([n_sample, len(lam)])
    sig = np.zeros([n_sample, len(lam)])
    for i, pos in enumerate(pars):
        m, s = model.model(lam, pos)
        mu[i, :] = m
        sig[i, :] = s
    return mu, np.abs(sig), pars
