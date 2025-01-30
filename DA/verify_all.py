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

import lim_utils

yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)

# load reconstructed Indexes
main_path = config['main_path'] + config['name'] + "/"
pic_path = main_path + "pic_noadj/"

if not os.path.exists(pic_path): os.makedirs(pic_path)

ens_ts_path = main_path + "recon/" + "indexes.nc"

reference_ts_path = config['Index_verify_path']
anom_period = config['anom_period']

ens_ts = xr.open_dataset(ens_ts_path)

show_period = config['Index_show_period']
show_period = slice(str(show_period[0]),str(show_period[1]))

index_infos = config['Indexs']

for index_name, index_info in index_infos.items():
    print(index_name)
    index = ens_ts[index_name]
    # ========================================================
    index_anom_adj = index.groupby('time.month') - index.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean() * 0
    cfr_ens_ts = cfr.EnsTS(value=index_anom_adj.values,time=cfr.utils.datetime2year_float(index.time.to_numpy()),value_name=index_name)
    # ==============================no compare==========================
    fig, ax = cfr_ens_ts.annualize().plot_qs()
    plt.savefig(pic_path + index_name + "no_compare_annulized.png",bbox_inches='tight')
    plt.show()
    # ========================================================
    continue
    print(index_info['verify'])
    if index_info['verify'] != "None" and index_info['verify'] != None:
        reference = xr.open_dataset(reference_ts_path + '/' + index_name + '.nc')[index_name]
        reference_anom_adj = reference.groupby('time.month') - reference.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean()
        cfr_reference = cfr.EnsTS(value=reference_anom_adj.values,time=reference.time.to_numpy(),value_name=index_name)
        setting_compare_period = index_info.get('compare_period')
        if setting_compare_period != None:
            compared_period = [int(setting_compare_period[0]),int(setting_compare_period[1])]
        else:
            compared_period = None
        # if index_info.get("time_reselution") != "year":
        #     fig,ax = cfr_ens_ts.compare(cfr_reference,timespan=compared_period).plot_qs()
        #     plt.savefig(pic_path + index_name + "compare.png",bbox_inches='tight', )
        #     plt.show()
        for sn in [1,4,7,10]:
            fig, ax = cfr_ens_ts.annualize(months=[sn]).compare(cfr_reference.annualize(months=[sn]),timespan=compared_period).plot_qs()
            plt.savefig(pic_path + index_name + f"all_{sn}.png",bbox_inches='tight')
            plt.show()
        
        

    



