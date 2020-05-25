rcat_vers=2_4  # Version of the rcat
rtype=RCAT_KIN # Use rcat values for kinematic quantities, not recomputed values
dist_err=0.1   # distance uncertainty to use for mocks
dly=+0.0       # shift in Ly-Lz selection line
flx=0.9        # select only |Lx| < flx * sqrt(Ly^2 + Lz^2)
max_rank=2     # only use stars with XFIT_RANK <= this
refit=false    # whether to refit the velocities

fdir=figures/V$rcat_vers/dly$dly
mkdir -p $fdir

# Fit velocities
if [ "${refit}" = true ]; then
  python fit_velocities.py --rcat_vers $rcat_vers --fit_leading --fit_trailing --fit_lm10 --fit_h3
fi

# Figure 1
python selection_lylz.py    --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir --ncol 3 \
                            --fractional_distance_error $dist_err --mag_cut --noisify_pms
# Figure 2
python metallicities.py     --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir --reweight
# Figure 3
python show_velocity_fit.py --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 4
python ely.py               --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 5
python vgsr_lambda_feh.py   --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 6
python mdf_by_vlam.py       --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 7
python x_lambda_mocks.py    --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 8
python x_lambda_selectmocks.py --dly $dly --flx $flx --max_rank=$max_rank \
                               --rcat_vers $rcat_vers --rcat_type $rtype \
                               --savefig --figure_dir $fdir \
                               --fractional_distance_error $dist_err --mag_cut --noisify_pms
# Figure 9
python quiver.py            --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir \
                            --fractional_distance_error $dist_err --mag_cut --noisify_pms

python franken_extrasig.py  --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir \
                            --fractional_distance_error $dist_err --mag_cut --noisify_pms

# Appendix
# Figure 10
python el_unc.py            --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
# Figure 11
python globulars.py         --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir

# --- Extras ---
python quiver.py            --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir \
                            --fractional_distance_error $dist_err --split --mag_cut
python vgsr_lambda_mocks.py --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
python dist_lambda_mocks.py --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
python beta_lambda_mocks.py --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --figure_dir $fdir
python lylz_feh.py          --dly $dly --flx $flx --max_rank=$max_rank \
                            --rcat_vers $rcat_vers --rcat_type $rtype \
                            --savefig --flx $flx --figure_dir $fdir
