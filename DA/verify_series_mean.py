"""
verify.py is verify the ens indexs
recon_sep_mean.py is verify the mean indexs

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

sys.path.append("/glade/derecho/scratch/zilumeng/SI_recon/LIM/")

# anom_adj = ['SIE']
anom_adj = []

import lim_utils

yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)
if config.get("MJob_name") is not None: # mul jobs
    main_path = config['Mpath'] + "/" + config['MJob_name'] + "/"
    MulJob = True
    # open the file config file for config
    with open(main_path + config['MJob_name'] + "_0" + "/config.yml", 'r') as stream: config = yaml.safe_load(stream)
else:
    main_path = config['main_path'] + config['name'] + "/"
    MulJob = False

# load reconstructed Indexes
pic_path = main_path + "pic1/"

if not os.path.exists(pic_path): os.makedirs(pic_path)

logger = slim.get_logger()
# logger.setLevel(logging.INFO)

# ens_ts_path = main_path + "recon/" + "indexes.nc"
ens_ts_path = main_path + "recon_sep1/" + "index_ens_mean.nc"

reference_ts_path = config['Index_verify_path']
anom_period = config['anom_period']

# ens_ts = xr.open_dataset(ens_ts_path)
logger.info("Load ens_ts from {}".format(ens_ts_path))
ens_ts = xr.open_dataset(ens_ts_path)
# ens_ts = ens_ts.sortby('ens_num')
# ens_ts = ens_ts.chunk({'ens_num':500})
# print("load ens_ts, shape: ",ens_ts)
logger.info("load ens_ts, shape: {}".format(ens_ts))

show_period = config['Index_show_period']
show_period = [int(show_period[0]),int(show_period[1])]

index_infos = config['Indexs']
for index_name, index_info in index_infos.items():
    logger.info(f"=========================Processing {index_name}=========================")
    index_pic_path = pic_path + index_name + "/"
    if not os.path.exists(index_pic_path): os.makedirs(index_pic_path)
    index = ens_ts[index_name]
    # ========================================================
    if index_name in anom_adj:
        index = index.groupby('time.month') - index.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean()
    cfr_ens_ts = cfr.EnsTS(value=index.values,time=cfr.utils.datetime2year_float(index.time.to_numpy()),value_name=index_name)
    # ==============================no compare==========================
    fig, ax = cfr_ens_ts.plot_qs()
    plt.savefig(index_pic_path + "seasonal.png",bbox_inches='tight')
    plt.show()
    fig, ax = cfr_ens_ts.annualize().plot_qs()
    plt.savefig(index_pic_path + "annual.png",bbox_inches='tight')
    plt.show()
    # ========================================================
    print(index_info['verify'] + "====================================")
    if index_info['verify'] != "None" and index_info['verify'] != None:
        logger.info(f"Comparing {index_name} with {index_info['verify']}")
        cfr_ens_ts = cfr_ens_ts
        reference = xr.open_dataset(reference_ts_path + '/' + index_name + '.nc')[index_name]
        reference_anom_adj = reference.groupby('time.month') - reference.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean()
        cfr_reference = cfr.EnsTS(value=reference_anom_adj.values,time=reference.time.to_numpy(),value_name=index_name)
        setting_compare_period = index_info.get('compare_period')
        if setting_compare_period != None:
            compared_period = [int(setting_compare_period[0]),int(setting_compare_period[1])]
        else:
            compared_period = None
        if index_info.get("time_reselution") != "year":
            fig,ax = cfr_ens_ts.compare(cfr_reference,timespan=compared_period).plot_qs(xlim=show_period)
            plt.savefig(index_pic_path + "compare_seasonal.png",bbox_inches='tight', )
            plt.show()
            for sn in [1,4,7,10]:
                fig, ax = cfr_ens_ts.annualize(months=[sn]).compare(cfr_reference.annualize(months=[sn]),timespan=compared_period).plot_qs(xlim=show_period)
                ax.set_title(f"Season {sn}")
                plt.savefig(index_pic_path + f"{index_name} compare_{sn}.png",bbox_inches='tight')
                plt.show()
        fig, ax = cfr_ens_ts.annualize().compare(cfr_reference.annualize(),timespan=compared_period).plot_qs(xlim=show_period)
        ax.set_title(f"{index_name} Annual")
        plt.savefig(index_pic_path + "compare_annual.png",bbox_inches='tight')
        plt.show()
        
        

    



