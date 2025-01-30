from typing import Hashable
from pandas.core.api import Index as Index
import xarray as xr
import slim
import numpy as np
import cfr
from xarray.core.dtypes import NA
import pandas as pd

AREA_PATH = "/glade/derecho/scratch/zilumeng/CMIP_LM/CCSM4/sh/area.nc"

def save_ens(data,coords,ens=False):
    pass


def recon_da_single(np_array,info,time,logger):
    domain = info['domain']
    if domain == "NH":
        lats=np.arange(-89,90,2)
        lats = lats[lats>=0]
    else:
        lats=np.arange(-89,90,2)
    lons=np.arange(0,360,2)
    coords = {"time":time,"ens_num":np.arange(np_array.shape[1]),"lat":lats,"lon":lons}
    da = xr.DataArray(np_array,coords=coords)
    logger.info("Reconstructed {} shape: {}, coords: {}".format(info['name'],da.shape,coords.keys()))
    return da


def recon_da_dict(np_dict,infos,time,ens_num,logger=None,ens=True):
    # time = cfr.utils.year_float2datetime(time.to_numpy())
    all_da = {}
    if logger is None: logger = slim.get_logger()
    for var_name in infos.keys():
        info = infos[var_name]
        domain = info['domain']
        if domain == "NH":
            lats=np.arange(-89,90,2)
            lats = lats[lats>=0]
        else:
            lats=np.arange(-89,90,2)
        lons=np.arange(0,360,2)
        # print(np_dict[var_name].shape)
        # print(lons.shape,lats.shape,time.shape)
        if ens:
            coords = {"time":time,"ens_num":np.arange(ens_num),"lat":lats,"lon":lons}
        else:
            coords = {"time":time,"lat":lats,"lon":lons}
        data = np_dict[var_name]
        if not ens and data.ndim == 4:
            data = np.mean(data,axis=1)
        elif not ens and data.ndim == 3:
            data = data
        da = xr.DataArray(data,coords=coords)
        all_da[var_name] = da
        logger.info("Reconstructed {} shape: {}, coords: {}".format(var_name,da.shape,coords.keys()))
    return all_da

def load_verify(verify_dict,all_verify_path,logger=None):
    if logger is None: logger = slim.get_logger()
    verify_save_dict = {}
    for var_name,verify_info in verify_dict.items():
        verify = xr.open_dataset(all_verify_path + verify_info['verify_name'])[verify_info['alias']]
        start_time = str(verify_info['verfiy_period'][0])
        end_time = str(verify_info['verfiy_period'][1])
        verify = verify.sel(time=slice(start_time,end_time))
        verify_save_dict[var_name] = verify
        logger.info("Loaded {} from {} to {}, shape: {}".format(var_name,start_time,end_time,verify.shape))
    return verify_save_dict


def verify_pattern_corr(all_da,verify_dict,logger=None):
    if logger is None: logger = slim.get_logger()
    corr_dict = {}
    for var in verify_dict.keys():
        verify = verify_dict[var]
        da = all_da[var]
        start_time = str(verify.time[0].dt.year) + "-" + str(verify.time[0].dt.month)
        end_time = str(verify.time[-1].dt.year) + "-" + str(verify.time[-1].dt.month)
        da = da.sel(time=slice(start_time,end_time))
        logger.info("Selected {} from {} to {}, da shape, verify shape: {}".format(var,start_time,end_time,da.shape,verify.shape))
        corr = slim.field_corr(da.to_numpy(),verify.to_numpy())
        corr_dict[var] = corr
        logger.info("Pattern correlation for {}: shape {}".format(var,corr.shape))
    return corr_dict


# def GMT_cal(x,ens=True):
#     if ens:
#         ens_dim = [0,1]
#     else:
#         ens_dim = [0]
#     Nan_mask = np.isnan(x).astype(int)
#     weights = np.cos(np.deg2rad(x.lat.to_numpy()))[:, None]
#     weights = np.repeat(weights,x.lon.size,axis=1)
#     if ens:
#         weights = np.repeat(weights[None,:,:],x.shape[ens_dim[1]],axis=0)
#         weights = np.repeat(weights[None,:,:,:],x.shape[ens_dim[0]],axis=0)
#     else:
#         weights = np.repeat(weights[None,:,:],x.shape[ens_dim[0]],axis=0)
#     weights = weights * (1 - Nan_mask) # mask nan
#     x_weighted = x * weights
#     x_weighted_mean = np.nansum(x_weighted,axis=(-1,-2)) / np.nansum(weights,axis=(-1,-2))
#     return x_weighted_mean

def GMT_cal(x):
    weights = np.cos(np.deg2rad(x.lat))
    weights.name = "weights"
    x_weighted = x.weighted(weights)
    x_weighted_mean = x_weighted.mean(dim=['lat','lon'])
    return x_weighted_mean


def IOD_cal(x):
    IODW = x.sel(lat=slice(-10,0),lon=slice(50,70)).mean(dim=['lat','lon'])
    IODE = x.sel(lat=slice(0,10),lon=slice(90,110)).mean(dim=['lat','lon'])
    IOD = IODW - IODE
    return IOD

def SIE_cal(x,mean_path,area_path=AREA_PATH):
    """
    Sea ice extent calculation
    """
    mean_sic = xr.open_dataset(mean_path)['sic'].sel(lat=slice(0,90))
    sum_x = x.groupby("time.month") + mean_sic.groupby("time.month").mean()
    area = xr.open_dataset(area_path)['cell_area'].sel(lat=slice(0,90))
    sum_area = xr.where(sum_x > 15,area,0)
    SIE = sum_area.sum(dim=['lat','lon'])
    return SIE




cal_function = {
    "Nino34": lambda x: x.sel(lat=slice(-5,5),lon=slice(190,240)).mean(dim=['lat','lon']),
    "NH-Temp": lambda x: GMT_cal(x.sel(lat=slice(0,90))),
    "SH-Temp": lambda x: GMT_cal(x.sel(lat=slice(-90,0))),
    "GMT": lambda x: GMT_cal(x),
    "IOD": IOD_cal,
    "GMOHC": lambda x: GMT_cal(x.sel(lat=slice(0,90))),
    "Nino1+2": lambda x: x.sel(lat=slice(-10,0),lon=slice(270,280)).mean(dim=['lat','lon']),
    "Nino3": lambda x: x.sel(lat=slice(-5,5),lon=slice(210,270)).mean(dim=['lat','lon']),
    "Nino4": lambda x: x.sel(lat=slice(-5,5),lon=slice(160,210)).mean(dim=['lat','lon']),
    "WPI": lambda x: x.sel(lat=slice(-10,10),lon=slice(120,150)).mean(dim=['lat','lon']),
    "IOBW": lambda x: x.sel(lat=slice(-20,20),lon=slice(40,100)).mean(dim=['lat','lon']),
    "SIE": SIE_cal,
}



def calculate_index(recon_dict,indexs_info,logger):
    indexes_dict = {}
    for index_name,index_info in indexs_info.items():
        var_need = index_info['var']
        function4cal = cal_function[index_name]
        if index_name == "SIE":
            function4cal = lambda x: cal_function[index_name](x,index_info['mean_path'])
        index_caled = function4cal(recon_dict[var_need])
        indexes_dict[index_name] = index_caled
        logger.info("Calculated index {} shape: {}".format(index_name,index_caled.shape))
        if True in np.isnan(index_caled.to_numpy()):
            logger.warning(f"NaN value in index {index_name}")
    return indexes_dict

def calculate_index_single(recon_da,indexs_info,var_name,logger,indexes_dict={}):
    for index_name,index_info in indexs_info.items():
        if index_info['var'] == var_name:
            function4cal = cal_function[index_name]
            if index_name == "SIE":
                function4cal = lambda x: cal_function[index_name](x,index_info['mean_path'])
            index_caled = function4cal(recon_da)
            logger.info("Calculated index {} shape: {}".format(index_name,index_caled.shape))
            if True in np.isnan(index_caled.to_numpy()):
                logger.warning(f"NaN value in index {index_name}")
            indexes_dict[index_name] = index_caled
    return indexes_dict
        

    



class EnTS:
    def __init__(self, data,time=None):
        if isinstance(data, xr.DataArray):
            self.data = data
        else:
            self.data = xr.DataArray(data)
        if self.data.ndim != 3:
            raise ValueError(f"Data should be 3D, not {self.data.shape}. Please check the shape of data.")
        


def independant_verify(pname, proxy, mask: bool ,recon,obs_name,logger, calib_period=(1850, 2000)):
    """
    indepentant verification for a single proxy
    pname: str, proxy name
    proxy: cfr.Proxy
    mask: bool, whether the proxy is masked
    recon: xarray.DataArray, the reconstruction
    obs_name: str, the name of the observation in the reconstruction (tas or tos etc.)
    logger: logger
    """
    detail = proxy.psm.calib_details
    attr_dict = {}
    attr_dict['name'] = pname
    attr_dict['seasonality'] = detail['seasonality']
    if mask:
        attr_dict['using'] = True
    else:
        attr_dict['using'] = False
    # =================== Get Recon of Proxy =====================
    ReconValueAnnual = cfr.utils.annualize(da=recon,months=detail['seasonality'])
    ReconValueAnnual['time'] = np.int32(ReconValueAnnual.time.dt.year)
    ReconDf = pd.DataFrame({"time":ReconValueAnnual.time.values,obs_name: ReconValueAnnual.values})
    Estimated = proxy.psm.model.predict(ReconDf)
    EstimatedDf = pd.DataFrame({"time":ReconDf.time, 'estimated':Estimated.values})
    ProxyDf = pd.DataFrame({"time":proxy.time, 'proxy':proxy.value})
    Df = ProxyDf.dropna().merge(EstimatedDf,on='time',how='inner')
    Df.set_index('time', drop=True, inplace=True)
    Df.sort_index(inplace=True)
    Df.astype(float)
    # =================== calculate corr and CE =====================
    masks = {'all': None,
        'in': (Df.index >= calib_period[0]) & (Df.index <= calib_period[1]), 
        'before': (Df.index < calib_period[0])}
    for mask_name, mask in masks.items():
        if mask is not None:
            Df_masked = Df[mask]
        else:
            Df_masked = Df
        if len(Df_masked) < 10:
            corr = np.nan
            ce = np.nan
        else:
            corr = Df_masked.corr().iloc[0,1]
            ce = cfr.utils.coefficient_efficiency(Df_masked.proxy.values,Df_masked.estimated.values)
        attr_dict[mask_name + '_corr'] = corr
        attr_dict[mask_name + '_ce'] = ce
        logger.info("Proxy {} {} Corr: {}, CE: {}".format(pname, mask_name, corr, ce))
    return attr_dict
