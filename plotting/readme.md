# Figure making code

Figures can be generated with `make_figs.sh`.  Some figures require that other scripts have been run first (`fit_velocities.py`, `lm10_seds.py`, `make_weight.py`)


- *Fig 1*: `selection_lylz.py`

   Sgr selection using Ly-Lz and comparing to LM10 and DL17

- *Fig 2*: `metallicities.py`

   [a/Fe] vs. [Fe/H] and [Fe/H] histogram

- *Fig 3*: `show_velocity_fit.py`

  Two panel plot, showing the fitted trend of velocity and velocity dispersion with Lambda

- *Fig 4*: `ely.py`

   2-panel E-Ly.  left All, right: low metallicity.  Sgr members maybe black and background in grey (background would be all giants above the SNR cut).  Show LM10 also?

- *Fig 5*: `vgsr_lambda_feh.py`

   3-panel V_GSR vs. Lambda in three metallicity slices, with diffuse/cold marked.

- *Fig 6*: `mdf_by_vlam.py`

   [Fe/H] histogram for several selections in V-Lam space. Lead/Trail x Cold/Diffuse

- *Fig 7*: `x_lambda_mocks.py`

   V_GSR and R_gc vs. Lambda for H3, LM10, and DL17. Color coded by Fe/H, and progenitor radius or t_unbound or DM/stars

- *Fig 8*: `x_lambda_selectmocks.py`

    As for Fig 7 but with a (basic) selection fn applied

- *Fig 9*: `franken_extrasig.py`

    Vgsr vs Lambda for H3 and LM10, with extra velocity dispersion added to LM10.

- *Fig 10*: `quiver.py`

   quiver plot

- *Fig A1*: `el_unc.py`

   Uncertainties in Ly-Lz and E-Ly for low metallicity stars.

- *Fig A2*: `globulars.py`

   Globular clusters in Ly-Lz and Vgsr, R_gc vs. Lambda
