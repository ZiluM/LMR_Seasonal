"""
reconstruct the data in single ens and calculate the index (used)
"""

import pickle
import numpy as np
import cfr
import da_utils
import xarray as xr
import recon_utils
import yaml 
import os,sys
import slim

sys.path.append("/glade/derecho/scratch/zilumeng/SI_recon/LIM/")

import lim_utils

yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)


main_path = config['main_path'] + config['name'] + "/"
config['main_path1'] = main_path
data_path = main_path + "data/"
logger = slim.get_logger(save_path=main_path + 'log_recon.txt',mode='w')

# ============= Lim Model config================

model_config_path = config['lim_config']
with open(model_config_path, 'r') as stream: lim_config = yaml.safe_load(stream)

_,eof_ls = lim_utils.load_pcs(lim_config['build_dir'],lim_config['vars_info'])

# ================== load data ==================

logger.info("Loading data from {}".format(data_path),)
whole_data = xr.open_mfdataset(data_path + "*.nc",combine='by_coords')['pcs'] # time, ens, pcs
logger.info("Data loaded, Data shape: {}".format(whole_data.shape))

whole_data['time'] = cfr.utils.year_float2datetime(whole_data.time.to_numpy())

# ================== decode data ==================

recon_path = main_path + "recon_sep/"
if not os.path.exists(recon_path): os.makedirs(recon_path)
indexs_info = config['Indexs']


ens_num = int(config['ens_num'])
# ens_num = 30 # rember to change


# ================= calculate the mean =====================
logger.info("Reconstructing ensemble mean")
recon = lim_utils.decoder_seq(whole_data.to_numpy()[:,:,:].mean(axis=1),eof_ls,lim_config['vars_info'],logger=logger,months=whole_data.time.dt.month.to_numpy())
recon_dict = {var_name:recon[i] for i,var_name in enumerate(lim_config['vars_info'].keys())}
all_da = recon_utils.recon_da_dict(recon_dict,lim_config['vars_info'],whole_data.time,1,logger=logger,ens=False)
for var_name,da in all_da.items():
    # save data
    ds = xr.Dataset({var_name:da})
    ds.to_netcdf(recon_path + var_name + "_mean.nc")
    logger.info("Reconstructed {} saved at: {}".format(var_name,recon_path + var_name + "_mean.nc"))

# ================= calculate the ensemble =====================
for ens_idx in range(ens_num):
    logger.info(" =====================Reconstructing ensemble {} ====================".format(ens_idx))
    recon = lim_utils.decoder_seq(whole_data.to_numpy()[:,[ens_idx],:],eof_ls,lim_config['vars_info'],logger=logger,months=whole_data.time.dt.month.to_numpy())
    recon_dict = {var_name:recon[i] for i,var_name in enumerate(lim_config['vars_info'].keys())}
    all_da = recon_utils.recon_da_dict(recon_dict,lim_config['vars_info'],whole_data.time,1,logger=logger,ens=True)
    # ======== save data =========
    # for var_name,da in all_da.items():
    #     da['ens_num'] = [ens_idx]
    #     # save data
    #     ds = xr.Dataset({var_name:da})
    #     ds.to_netcdf(recon_path + var_name + "_ens{}.nc".format(ens_idx))
    #     logger.info("Reconstructed {} saved at: {}".format(var_name,recon_path + var_name + "_ens{}.nc".format(ens_idx)))
    # ======== calculate index =========
    indexes_dict = recon_utils.calculate_index(all_da,indexs_info,logger=logger)
    da_indexes_dict = { var_name: xr.DataArray(np.array(indexes_dict[var_name]),coords={'time':whole_data.time,"ens_num":[ens_idx]}) for var_name in indexes_dict.keys()}
    # save indexes
    xr.Dataset(da_indexes_dict).to_netcdf(recon_path + "index_ens{}.nc".format(ens_idx))
    logger.info("Indexes saved at: {}".format(recon_path + "index_ens{}.nc".format(ens_idx)))


    
    
    
    
