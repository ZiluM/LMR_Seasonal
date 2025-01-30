import yaml 
import os,sys
import slim
import pickle
import numpy as np
import cfr
import da_utils
import pandas as pd
import xarray as xr
import functools
from da_utils import Linear

sys.path.append("/glade/derecho/scratch/zilumeng/SI_recon/LIM/")

import lim_utils

yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)

main_path = config['main_path'] + config['name'] + "/"
config['main_path1'] = main_path

# ================== Create folder and save config ==================
if not os.path.exists(main_path):
    os.makedirs(main_path)
else:
    print("The folder already exists, please change the name in the yml file.")
if not os.path.exists(main_path + "pic/"): os.makedirs(main_path + "pic/")
if not os.path.exists(main_path + "data/"): os.makedirs(main_path + "data/")
os.system("cp {} {}".format(yml_path, main_path + "config.yml"))
logger = slim.get_logger(save_path=main_path + 'log.txt',mode='w')

# ============= Lim Model config================ 
model_config_path = config['lim_config']
with open(model_config_path, 'r') as stream: lim_config = yaml.safe_load(stream)
LIM_save_path = lim_config['main_path'] + lim_config['name'] + "/" + "model.pkl"
with open(LIM_save_path, 'rb') as f: LIM = pickle.load(f)
logger.info("Model loaded from: {}, Model type {}".format(LIM_save_path,type(LIM)))
LIM.mtype = lim_config['type']
# ============= Load EOF and PC ================

pc_ls,eof_ls = lim_utils.load_pcs(lim_config['build_dir'],lim_config['vars_info'])
logger.info("pcs_ls shape: {}".format(pc_ls.shape))
logger.info("eof_ls shape: {}".format(len(eof_ls)))

# create ensemble pool

ens_num = config['ens_num']

ensemble_pool = [pc_ls[i::4][:ens_num] for i in range(4)] # 4, ens_num, pc_num
ensemble_pool = np.array(ensemble_pool) # 4, ens_num, pc_num

ensemble_time_pool = []

logger.info("Ensemble Pool shape: {}".format(ensemble_pool.shape))
# save_ensemble = np.copy(ensemble_pool[[0]])

# ============= Load OBS ========================

OBS_JOB = cfr.ReconJob()
OBS_JOB.load(config['obs_path'])
# filter the ptype
if config.get('ptype',None) is not None:
    OBS_JOB.proxydb = OBS_JOB.proxydb.filter(by='ptype', keys=config['ptype'])
    logger.info("=========== ptype filter applied: {} =============".format(config['ptype']))
else:
    logger.info("No ptype filter applied.")

# filter the pid
if config.get('pid',None) is not None and config.get('use_single_proxy') is True:
    OBS_JOB.proxydb = OBS_JOB.proxydb.filter(by='pid', keys=config['pid'])
    logger.info("=========== pid filter applied: {} =============".format(config['pid']))


OBS_pdb = OBS_JOB.proxydb.filter(by='tag',keys=["calibrated"])
logger.info("OBS loaded from: {}".format(config['obs_path']))
logger.info("OBS proxy type and number: {}".format(OBS_pdb.type_dict))

# ============= Mask OBS ========================

pdb_using, pdb_vatify,mask_array = da_utils.mask_obs(OBS_pdb,config['mask_rate'],int(config['mask_seed']),logger)
np.save(main_path + "mask_array.npy",mask_array)
logger.info("Mask Array saved at: {}".format(main_path + "mask_array.npy"))

# ============= select the updata strategy: update_ensemble_function ========================
# this is for the timing and enkf time localization; for example, how the proxy influence the different time steps
if config.get('update_strategy',None) is not None:
    update_strategy = config['update_strategy']
    if update_strategy == "endofyear": # proxy only be used at the end of year, but it will influence the whole year
        update_ensemble_function = da_utils.update_ensemble_endofyear
        logger.info("Update strategy: endofyear")
    elif update_strategy == "justend": # proxy only be used at the end of its season, and it will only influence this season
        update_ensemble_function = da_utils.update_ensemble_just_end
        logger.info("Update strategy: justend")
    elif update_strategy == "errorMatrix": # use the error matrix to update the ensemble
        error_matrix_path = config['obs_path'] + "/" + "error_matrix.pkl"
        try:
            error_matrix = pd.read_pickle(error_matrix_path)
        except:
            raise ValueError("Error matrix not found at: {}".format(error_matrix_path))
        update_ensemble_function = functools.partial(da_utils.update_ensemble_endofyear_error_matrix,error_matrix=error_matrix)
        logger.info("Update strategy: Error Matrix, error matrix loaded from: {}, shape: {}".format(error_matrix_path, error_matrix.shape))
    elif update_strategy == "default": # use the default update strategy, the proxy will influence the season which is calibrated, and updated at the end of its season
        update_ensemble_function = da_utils.update_ensemble
        logger.info("Update strategy: default")
    else:
        logger.info("Update strategy not found: {}, must be in ['endofyear','justend','errorMatrix']".format(update_strategy))
        raise ValueError("Update strategy not found: {}, must be in ['endofyear','justend','errorMatrix']".format(update_strategy))
    
        
else:
    update_ensemble_function = da_utils.update_ensemble
    logger.info("Update strategy: default")

# ============= select the update function: update_function ========================

if config.get('update_function',None) is not None:
    update_function_name = config['update_function']
    if update_function_name == "default":
        logger.info("Update function: default")
        update_function = da_utils.enkf_update_array
    elif update_function_name == "EnKF":
        logger.info("Update function: EnKF")
        update_function = da_utils.enkf_perturbed_observation
    elif update_function_name == "EnSRF":
        logger.info("Update function: EnSRF")
        update_function = da_utils.enkf_update_array
    else:
        logger.info("Update function not found: {}, must be in ['default','EnKF','EnSRF']".format(update_function_name))
        raise ValueError("Update function not found: {}, must be in ['default','EnKF','EnSRF']".format(update_function_name))
else:
    update_function = da_utils.enkf_update_array
    logger.info("Update function: default")


# ============= DA start ========================
dt_month = 1 / 12


restart = bool(config['restart'])
# if restart:
#     restart_time = config['restart_time']
#     if restart == "latest":
#         reconst_data = xr.open_mfdataset(main_path + "data/*.nc",combine='by_coords')['pcs']
#         start_year = int(reconst_data.time[-1].dt.year)
#         start_month = int(reconst_data.time[-1].dt.month)
#         start_time = start_year + dt_month * (start_month - 1  + 0.5)
#         current_time = start_time
#         ensemble_time_pool = [current_time - 3 * i * dt_month for i in range(4)][::-1]
#         ensemble_pool = reconst_data[-4:].to_numpy() # 4, ens_num, pc_num
        # start_step = 
# else:
start_time = config['start_time'] 
start_year,start_month = int(start_time.split("-")[0]),int(start_time.split("-")[1])
start_time = start_year + dt_month * (start_month - 1  + 0.5)
current_time = start_time
ensemble_time_pool = [current_time - 3 * i * dt_month for i in range(4)][::-1]
start_step = 0

logger.info("Start DA at time {}".format(config['start_time']))

logger.info("Start time: {}".format(da_utils.float2inttime(start_time)))
logger.info("Ensemble time pool: {}".format([da_utils.float2inttime(i) for i in ensemble_time_pool]))

# ================== init save pool =====================

save_pool = []
save_time = []

all_saved_length = 0

# ================== DA loop =====================

for Da_t in range(start_step,config['da_length'],config['da_step']):
    current_time = start_time + Da_t * dt_month
    logger.info("DA step {}, current time {}".format(Da_t // config['da_step'],da_utils.float2inttime(current_time)))
    current_year,current_month = da_utils.float2inttime(current_time)
    # ============= Update step ========================
    ensemble_pool = update_ensemble_function(ensemble_pool,eof_ls,lim_config,pdb_using,current_time,logger,pseudo_label=config['pseudo'],update_func=update_function)
    # ============= Forecast step ========================
    # forecasted = LIM.noise_integration(ensemble_pool[-1],length=2,length_out_arr=np.zeros((2,ens_num,ensemble_pool.shape[-1])))[1:]
    forecasted = da_utils.mul_forecast(LIM,ens_num=ens_num,ensemble_pool=ensemble_pool,current_month=current_month,logger=logger)

    logger.info("Forecasted shape: {}, forecasted_var {:.3f}".format(forecasted.shape,forecasted.std(axis=-2).mean()))
    # ============= update ensemble pool for the next step ========================
    save_pool.append(ensemble_pool[0])
    # save_time.append(current_time - 3 * 3 * dt_month)
    save_time.append(ensemble_time_pool[0])

    ensemble_pool = np.concatenate([ensemble_pool[1:],forecasted],axis=0) # 4, ens_num, pc_num
    ensemble_time_pool = ensemble_time_pool[1:] + [current_time + 3 * dt_month]
    if Da_t % config['save_step'] == 0 and Da_t != 0:
        save_pool,save_time,saved_length = da_utils.save_da_result(save_pool,save_time,main_path,logger)
        all_saved_length += saved_length

# ============= Save the final result ========================
save_pool += [ensemble_pool[i] for i in range(0,3)]
save_time += [ensemble_time_pool[i] for i in range(0,3)]

save_pool,save_time,saved_length = da_utils.save_da_result(save_pool,save_time,main_path,logger)

all_saved_length += saved_length

logger.info("All saved length: {}".format(all_saved_length))


    



