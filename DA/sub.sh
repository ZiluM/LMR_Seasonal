conda activate cfr

cd /glade/derecho/scratch/zilumeng/SI_recon/DA
config_path=/glade/u/home/zilumeng/SI_recon/DA/config/CCSM4_noeofOBS_alltas.yml


python3 da.py $config_path
python3 recon_sep.py $config_path
python3 verify_series_mean.py $config_path
python3 verify_pattern.py $config_path



