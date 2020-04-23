rcat_vers=2_4
dist_err=0.1
dly=-0.5
refit=false

fdir=figures/V$rcat_vers/dly$dly
mkdir -p $fdir

# Fit velocities
if [ "${refit}" = true ]; then
  python fit_velocities.py --rcat_vers $rcat_vers --fit_leading --fit_trailing --fit_lm10 --fit_h3
fi

python selection_lylz.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir --ncol 3 \
                         --fractional_distance_error $dist_err --mag_cut --noisify_pms
python metallicities.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python show_velocity_fit.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python ely.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python vgsr_lambda_feh.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python mdf_by_vlam.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python x_lambda_mocks.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python x_lambda_selectmocks.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir \
                               --fractional_distance_error $dist_err --mag_cut --noisify_pms
python quiver.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir \
                 --fractional_distance_error $dist_err --mag_cut --noisify_pms

# Appendix
python el_unc.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python globulars.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir

# extras
python quiver.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir \
                 --fractional_distance_error $dist_err --split --mag_cut
python vgsr_lambda_mocks.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python dist_lambda_mocks.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python beta_lambda_mocks.py --dly $dly --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python lylz_feh.py --savefig --dly $dly --rcat_vers $rcat_vers --figure_dir $fdir