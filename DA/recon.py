import yaml 
import os,sys
import slim
import pickle
import numpy as np
import cfr
import da_utils
import xarray as xr
import recon_utils

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
recon = lim_utils.decoder_seq(whole_data.to_numpy(),eof_ls,lim_config['vars_info'],logger=logger)
recon_dict = {var_name:recon[i] for i,var_name in enumerate(lim_config['vars_info'].keys())}

all_da = recon_utils.recon_da_dict(recon_dict,lim_config['vars_info'],whole_data.time,whole_data.ens_num.size,logger=logger,ens=True)

# ================== save mean ==================
recon_path = main_path + "recon/"
if not os.path.exists(recon_path): os.makedirs(recon_path)



# save data
for var_name in all_da.keys():
    da = all_da[var_name]
    ds = xr.Dataset({var_name:da.mean(dim='ens_num')})
    ds.to_netcdf(recon_path + var_name + ".nc")
    logger.info("Reconstructed {} saved at: {}".format(var_name,recon_path + var_name + ".nc"))



# ================== Calculate Index ==================

indexs_info = config['Indexs']

indexes_dict = recon_utils.calculate_index(all_da,indexs_info,logger=logger)

da_indexes_dict = { var_name: xr.DataArray(np.array(indexes_dict[var_name]),coords={'time':whole_data.time,"ens_num":np.arange(whole_data.ens_num.size)}) for var_name in indexes_dict.keys()}


# ================== save index ==================
xr.Dataset(da_indexes_dict).to_netcdf(recon_path + "indexes.nc")
logger.info("Indexes saved at: {}".format(recon_path + "indexes.nc"))







