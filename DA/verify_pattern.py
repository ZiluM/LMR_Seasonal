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
import cartopy.crs as ccrs

sys.path.append("/glade/derecho/scratch/zilumeng/SI_recon/LIM/")


anom_adj = ['SIE']

import lim_utils

yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)

if config.get("MJob_name") is not None: # mul jobs
    main_path = config['Mpath'] + "/" + config['MJob_name'] + "/"
    MulJob = True
    # open the file config file for config
    with open(main_path + config['MJob_name'] + "_0" + "/config.yml", 'r') as stream: config = yaml.safe_load(stream)
# load reconstructed Indexes
else:
    main_path = config['main_path'] + config['name'] + "/"
    MulJob = False
pic_path = main_path + "pic1/"

pattern_pic_path = main_path + "pattern1/"

if not os.path.exists(pattern_pic_path): os.makedirs(pattern_pic_path)

# recon_path = main_path + "recon/"
recon_path = main_path + "recon_sep1/"

anom_period = config['anom_period']

pattern_info = config['Patterns']

for var_name, info in pattern_info.items():
    var_pattern_path = pattern_pic_path + var_name + "/"
    if not os.path.exists(var_pattern_path): os.makedirs(var_pattern_path)
    compare_period = info['compare_period']
    compare_period = slice(str(compare_period[0]), str(compare_period[1]))
    # recon_res = xr.open_dataset(recon_path + var_name + ".nc")[var_name].loc[compare_period]
    recon_res = xr.open_dataset(recon_path + var_name + "_mean.nc")[var_name].loc[compare_period]
    recon_res = recon_res.groupby('time.month') - recon_res.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean('time')
    print(recon_res)
    # ==========================load real==============================
    real_path = config['Pattern_verify_path'] + info['verify']
    real = xr.open_dataset(real_path)[info['alias']].loc[compare_period]
    print(real)
    real = real.groupby('time.month') - real.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean('time')
    if info.get("domain") == "NH":
        real = real.sel(lat=slice(0, 90))
        proj = "NorthPolarStereo"
        recon_res = recon_res.sel(lat=slice(0, 90))
        # re-climatology
        recon_clim = xr.open_dataset(info['mean_path'])[var_name].sel(lat=slice(0, 90))
        recon_res = recon_res.groupby('time.month') + recon_clim.groupby('time.month').mean()
        recon_res = xr.where(recon_res > 100, 100, recon_res)
        recon_res = xr.where(recon_res < 0, 0,recon_res)
        recon_res = recon_res.groupby('time.month') - recon_res.loc[str(anom_period[0]):str(anom_period[1])].groupby('time.month').mean('time')
        # recon_res = xr.where(recon_res.lat > 85, np.nan, recon_res)
        recon_res.loc[dict(lat=slice(85, 90))] = np.nan
        print(recon_res,real)
    else: 
        proj = "Robinson"
    cfr_real = cfr.ClimateField(real)
    cfr_recon = cfr.ClimateField(recon_res)
    # ==========================plot==============================
    compared = cfr_recon.compare(cfr_real,interp=False)
    # lat = compared.da.lat
    # weights = np.sqrt(np.cos(np.deg2rad(lat)))
    # weights.name = 'weights'
    # da_weighted = compared.da.weighted(weights)
    # da_weighted_mean = da_weighted.mean(dim=('lat','lon'))
    da_weighted_mean = recon_utils.GMT_cal(compared.da)
    compared.plot_kwargs['title'] = "Correlation (mean = {:.2f})".format(da_weighted_mean.values[0])
    fig,ax = compared.plot(projection=proj)
    if info.get("domain") == "NH": ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
    plt.savefig(var_pattern_path  + "all.png")
    plt.show()
    # ========================== annual correlation ==============================
    # if info.get("domain") == "NH":
    #     sn = [6,7,8]
    # else:
    #     sn = None
    for sn,sn_name in zip([list(np.arange(1,13)),[1],[4],[7],[10]],["ANN","DJF","MAM","JJA","SON"]):
        ann_cfr = cfr_recon.annualize(sn)
        ann_cfr_real = cfr_real.annualize(sn)
        ann_compared = ann_cfr.compare(ann_cfr_real,interp=False,)
        # lat = ann_compared.da.lat
        da_weighted_mean = recon_utils.GMT_cal(ann_compared.da)
        ann_compared.plot_kwargs['title'] = "{} Correlation (mean = {:.2f})".format(sn_name,da_weighted_mean.values[0])
        if info.get("domain") == "NH": 
            fig,ax = ann_compared.plot(projection=proj,mode="mesh")
            ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
        else:
            fig,ax = ann_compared.plot(projection=proj)

        plt.savefig(var_pattern_path + sn_name + "_annual.png")
        plt.show()


    



