rcat_vers=2_4
fdir=figures/V$rcat_vers
dist_err=0.1

mkdir -p $fdir

python selection_lylz.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python metallicities.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python ely.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python vgsr_lambda_feh.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python mdf_by_vlam.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python vgsr_lambda_mocks.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python dist_lambda_mocks.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir
python beta_lambda_mocks.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir --fractional_distance_error $dist_err
python x_lambda_selectmocks.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir --fractional_distance_error $dist_err
python quiver.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir --fractional_distance_error $dist_err
python quiver.py --savefig --rcat_vers $rcat_vers --figure_dir $fdir --fractional_distance_error $dist_err --split
