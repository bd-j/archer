#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as pl

try:
    from scipy.special import logsumexp
except(ImportError):
    from scipy.special import logsumexp


class Model:

    def __init__(self, alpha_range, beta_range, pout_range):
        """
        :param alpha_range:
            min and max for alpha, array_like, shape (2, alpha_order)
            :math:`vmu = \sum_i \alpha_i \Lambda^i`

        :param beta_range:
            min and max for beta, array_like, shape (2, alpha_order)
            :math:`vsig = \sum_i \beta_i \Lambda^i`

        :param pout_range:
            min and max for pout, array_like, shape (2,)

        """
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.pout_range = pout_range
        self.vmu_bad = -100
        self.vsig_bad = 200

    def pos_to_params(self, pos):
        na, nb = self.alpha_order, self.beta_order
        self.alpha = pos[:na]
        self.beta = pos[na: (na + nb)]
        self.pout = pos[-1]

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
        return np.hstack([a, b, pout])

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
        return self.beta_order + self.alpha_order + 1


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


if __name__ == "__main__":

    # --- Set Priors ---
    alpha_range = np.array([[100., -10., -0.2],
                            [1000., 0.1, 0.2]])
    beta_range = np.array([[-50., 0.],
                           [50., 1.0]])
    pout_range = np.array([0, 0.1])

    # --- Instantiate model ---
    model = Model(alpha_range=alpha_range, beta_range=beta_range,
                  pout_range=pout_range)
    model.set_data(lam, vel)

    # --- Fit ---
    from dynesty import DynamicNestedSampler as Sampler
    dsampler = Sampler(model.lnprob, model.prior_transform, model.ndim)
    dsampler.run_nested()
    dresults = dsampler.results
