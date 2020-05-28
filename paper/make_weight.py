#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from h3py.analysis.weighting import compute_total_weight, show_weights


if __name__ == "__main__":

    import os, time, sys
    from astropy.io import fits
    from archer.seds import Isochrone, FILTERS
    from archer.config import parser, rectify_config, plot_defaults
    _ = plot_defaults(pl.rcParams)

    try:
        parser.add_argument("--test", action="store_true")
        parser.add_argument("--test_feh", type=float, default=-1.0)
        parser.add_argument("--snr_limit", type=float, default=3.0)
        parser.add_argument("--limit_band", type=str, default="r",
                            help=("band for computing limiting snr, 'r' | 'g'"))
        parser.add_argument("--use_ebv", action="store_true")
        parser.add_argument("--use_afe", action="store_true")
        parser.add_argument("--use_age", action="store_true")
    except:
        pass

    config = rectify_config(parser.parse_args())

    filters = np.array(["PS_r", "PS_g", "PS_z",
                        "SDSS_u", "SDSS_g", "SDSS_r",
                        "WISE_W1", "WISE_W2"])
    iso = Isochrone(mistfile=config.mistiso, nnfile=config.nnfile, filters=filters)

    # test the mag weight code
    if config.test:
        fig, ax = show_weights(feh=config.test_feh, iso=iso)
        pl.ion()
        pl.show()
        sys.exit()

    rcat = fits.getdata(config.rcat_file)
    pcat = fits.getdata(config.pcat_file)

    # output wcat name
    outname = config.rcat_file.replace("rcat", "wcat")
    options = {"ebv": config.use_ebv, "alpha": config.use_afe, "age": config.use_age}
    tag = "_".join([k for k in options.keys() if options[k]])
    outname = outname.replace(".fits", "[{}_PS{}].fits".format(tag, config.limit_band))
    print(outname)
    assert (not os.path.exists(outname))

    # select the stars to calculate weights for
    from make_selection import good_select
    good = good_select(rcat, extras=False, snr_limit=config.snr_limit)

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
