# LM10

* `lambda` [deg] Sgr longitude (LM convention, (l, b)pole ~ (274, −14).
* `beta`   [deg] Sgr latitude (LM convention, (l, b)pole ~ (274, −14).
* `ra`     [deg] Right Ascencion
* `dec`    [deg] Declination
* `l`      [deg] Galactic longitude
* `b`      [deg] Galactic latitude
* `xgc`    [kpc] Galactocentric (kpc)  left-handed?
* `ygc`    [kpc] Galactocentric (kpc)  left-handed?
* `zgc`    [kpc] Galactocentric (kpc)  left-handed?
* `xsun`   [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda_sun system.
* `ysun`   [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda_sun system.
* `zsun`   [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda_sun system.
* `x4`     [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda4 system.
* `y4`     [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda4 system.
* `z4`     [kpc] Right-handed heliocentric Cartesian coordinate in the Lambda4 system.
* `u`      [?] Galactic U velocity (d(X_GC)/dt).
* `v`      [?] Galactic V velocity (d(Y_GC)/dt).
* `w`      [?] Galactic W velocity (d(Z_GC)/dt).
* `dist`   [kpc] heliocentric distance
* `vgsr`   [km/s] Radial velocity relative to the Galactic Standard of Rest (NB: GSR, not LSR).
* `mul`    [mas/yr] (cos(b)) Proper motion along galactic longitude, heliocentric frame
* `mub`    [mas/yr] Proper motion along galactic latitude, heliocentric fram
* `mua`    [mas/yr] (cos(dec)) Proper motion along right ascension, heliocentric frame
* `mud`    [mas/yr] Proper motion along declination, heliocentric frame
* `Pcol`   [int] "Color", or unbound era of simulated particle
* `Lmflag` [int] -1 for trailing arm debris, +1 for leading arm debris

* `Estar` [?] Order-sorted internal energy (kinematic+potential) in the initial satellite (see LM10).
* `Popn`  [int] Stellar population to which each particles is assigned
* `Fe/H`  [dex] Metallicity
* `Age`   [Gyr] Age in Gyr
* `tub`   [Gyr] Time before present (in Gyr) at which particle became unbound from Sgr
* `J-K`:  [mag] J-K color assigned
* `Kabs`  [mag] Absolute K magnitude
* `Kapp`  [mag] Apparent K magnitude


# DL17
Same for stars and dark matter particles

* `id`    [int] 0 for stars, 1 for dark matter  (id = 'is_dark')
* `lat`   [deg] galactic latitude
* `long`  [deg] galactic longitude
* `RA`    [deg] Right Ascencsion
* `dec`   [deg] Declination
* `vlos`  [km/s] Radial velocity relative to the Galactic Standard of Rest
* `dist`  [kpc] Heliocentric distance?
* `pmRA`  [mas/yr] proper motion (solar reflex motion corrected)
* `pmdec` [mas/yr] proper motion (solar reflex motion corrected)


# R18



# KSEGUE
relevant columns only, I think from original sources.


* `SDSS`
* `RAJ2000`
* `DEJ2000`
* `HRV`       [km/s]  Heliocentric radial velocity
* `e_HRV`     [km/s]
* `Teff`
* `logg`
* `DMpeak`    [mag] mode of the distance modulus PDF, see Xue et al 2014
* `DM05`
* `DM16`
* `DM50`
* `DM84`
* `DM95`
* `eDM`
* `rmag_lc`
* `e_rMag`
* `Dist`
* `e_Dist`
* `gaia.ra`
* `gaia.dec`
* `gaia.parallax`
* `gaia.parallax_error`
* `gaia.parallax_over_error`
* `gaia.pmra`
 'gaia.pmra_error',
 'gaia.pmdec',
 'gaia.pmdec_error',
 'gaia_dmatch',
 'FeH',
 'FeH_err',


