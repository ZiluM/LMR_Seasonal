"""
verify the assimilated proxies and unassimilated proxies

Zilu Meng; 2024-May-21
"""

import yaml 
import os,sys
import slim
import pickle
import numpy as np
import cfr
import da_utils
import xarray as xr
import recon_utils
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor



yml_path = sys.argv[1]

calib_period=(1850, 2000)



# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)

main_path = config['main_path'] + config['name'] + "/"

# ================== Init Logger ==================
logger = slim.get_logger(save_path=main_path + 'log_IndpVerify.txt',mode='w')

# =================== Load Recon =====================
recon_path = main_path + "recon_sep1/"
ReconDict = {}
for var_name in ['tos','tas']:
    ReconDict[var_name] = xr.open_dataset(recon_path + var_name + "_mean.nc")[var_name]
    
# =================== Load Observation =====================

OBS_JOB = cfr.ReconJob()
OBS_JOB.load(config['obs_path'])
OBS_pdb = OBS_JOB.proxydb.filter(by='tag',keys=["calibrated"])


# Load Mask

mask_array = np.load(main_path + "mask_array.npy")
logger.info("Mask loaded {}, mask rate {}".format(mask_array.shape, mask_array.mean()))

# =================== Load Proxy =====================

all_corr_ce = {'used':[],'unused':[]}
Df_collection = [] # list to save all proxy's attributes

with ProcessPoolExecutor(max_workers=8) as executor:
    for i, (pname, proxy) in enumerate(OBS_pdb.records.items()):
        
        # pname, proxy = list(OBS_pdb.records.items())[0]
        lat,lon = proxy.lat,proxy.lon
        nlat, nlon = da_utils.get_index4proxy(lat,lon) # get the 
        obs_name = proxy.psm.calib_details['df'].columns[1]
        logger.info("Proxy ({}) {} loaded".format(i,pname))
        recon = ReconDict[obs_name].isel(lat=nlat,lon=nlon)
        future = executor.submit(recon_utils.independant_verify,
        pname=pname,
        proxy=proxy,
        mask=mask_array[i],
        recon=recon,
        obs_name=obs_name,
        logger=logger,
        )
        Df_collection.append(future)
    # =================== Get Recon of Proxy =====================
    Df_collection = [future.result() for future in Df_collection]

Df_collection = pd.DataFrame(Df_collection)
logger.info("All proxies verified,\n {}".format(Df_collection))
# =================== Save Df =====================

Df_collection.to_csv(main_path + "Proxy_verify.csv")
logger.info("Proxy_verify.csv saved in {}".format(main_path))





