import cfr
import numpy as np
import slim
import os,sys
import pickle
import xarray as xr
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
sys.path.append("/glade/derecho/scratch/zilumeng/SI_recon/LIM/")
import lim_utils
import yaml
import pandas as pd
import functools
from cfr import psm
import statsmodels.formula.api as smf

class Linear:
    ''' A PSM that is based on univariate linear regression.

    Args:
        pobj (cfr.proxy.ProxyRecord): the proxy record object
        climate_required (cfr.climate.ClimateField): the required climate field object for running this PSM
    '''
    def __init__(self, pobj=None, climate_required=['tas']):
        self.pobj = pobj
        self.climate_required = climate_required

    def calibrate_subAnn(self, calib_period=None, nobs_lb=25, metric='fitR2adj',
        season_list=[list(range(1, 13))], exog_name=None, **fit_args):
        if exog_name is None:
            exog_name = f'obs.{self.climate_required[0]}'

        exog = self.pobj.clim[exog_name]

        if type(season_list[0]) is not list:
            season_list = [season_list]

        score_list = []
        mdl_list = []
        df_list = []
        sn_list = []
        exog_colname = exog_name.split('.')[-1]

        # for sn in season_list:
        # print(exog.__dict__)
        exog_ann = exog.da.resample(time='QS-DEC').mean()
        df_exog = pd.DataFrame({'time': exog_ann.time.values, exog_colname: exog_ann.values})
        df_proxy = pd.DataFrame({'time': year_float2datetime(self.pobj.time), 'proxy': self.pobj.value})
        # print(df_proxy,df_exog)
        df = df_proxy.dropna().merge(df_exog.dropna(), how='inner', on='time')
        print(len(df))
        # print(df)
        df.set_index('time', drop=True, inplace=True)
        df.sort_index(inplace=True)
        df.astype(float)
        if calib_period is not None:
            mask = ( datetime2year_float(df.index) >=calib_period[0]) & ( datetime2year_float(df.index)<=calib_period[1])
            df = clean_df(df, mask=mask)

        formula_spell = f'proxy ~ {exog_colname}'
        nobs = len(df)
        if nobs < nobs_lb:
            print(f'The number of overlapped data points is {nobs} < {nobs_lb}. Skipping ...')
        else:
            mdl = smf.ols(formula=formula_spell, data=df).fit(**fit_args)
            fitR2adj =  mdl.rsquared_adj,
            mse = np.mean(mdl.resid**2),
            score = {
                'fitR2adj': fitR2adj,
                'mse': mse,
            }
            score_list.append(score[metric])
            mdl_list.append(mdl)
            df_list.append(df)
            sn_list.append('SubAnn')

        if len(score_list) > 0:
            opt_idx_dict = {
                'fitR2adj': np.argmax(score_list),
                'mse': np.argmin(score_list),
            }

            opt_idx = 0
            opt_mdl = mdl_list[opt_idx]
            opt_sn = sn_list[opt_idx]

            calib_details = {
                'df': df_list[opt_idx],
                'nobs': opt_mdl.nobs,
                'fitR2adj': opt_mdl.rsquared_adj,
                'PSMresid': opt_mdl.resid,
                'PSMmse': np.mean(opt_mdl.resid**2),
                'SNR': np.std(opt_mdl.predict()) / np.std(opt_mdl.resid),
                'seasonality': opt_sn,
            }

            self.calib_details = calib_details
            self.model = opt_mdl
        else:
            self.calib_details = None
            self.model = None



# forecasted = LIM.noise_integration(ensemble_pool[-1],length=2,length_out_arr=np.zeros((2,ens_num,ensemble_pool.shape[-1])))[1:]

def wrap_noise_integration(lim,ensemble_pool_forecast,seed,current_month,logger=None):
    length_out_arr = np.zeros((2,ensemble_pool_forecast.shape[0],ensemble_pool_forecast.shape[1]))
    if lim.mtype == 'LIM':
        res = lim.noise_integration(ensemble_pool_forecast,length=2,length_out_arr=length_out_arr,seed=seed)[1:]
    elif lim.mtype == 'CSLIM':
        month_dc = {1:1,4:2,7:3,10:4}
        res = lim.noise_integration(ensemble_pool_forecast,month0=month_dc[current_month],length=1,seed=seed)[1:]
    elif lim.mtype == 'SNLIM':
        month_dc = {1:1,4:2,7:3,10:4}
        res = lim.noise_integration(ensemble_pool_forecast,length=2,month0=month_dc[current_month],seed=seed)[1:]
    # if logger is not None:
        # logger.info("Forecasted shape: {}, forecasted_var {:.3f}".format(res.shape,res.std(axis=-2).mean()))
    return res

def mul_forecast(lim,ens_num,ensemble_pool,current_month,logger=None):
    # length = 2
    single_ens_num = 100
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_list = []
        # lim1 = deepcopy(lim)
        for i in range(0,ens_num,single_ens_num):
            selected_ens = ensemble_pool[-1,i:i+single_ens_num]
            # print()
            # if logger is not None: logger.info("Forecasting ensemble id: {}".format(id(selected_ens)))
            future = executor.submit(wrap_noise_integration,lim,selected_ens,seed=np.random.randint(0,1000000),logger=logger,current_month=current_month)
            future_list.append(future)
    future_list = [future.result() for future in future_list]
    forecasted = np.concatenate(future_list,axis=1)
    return forecasted

            


def enkf_perturbed_observation(Xb, obvalue, Ye, ob_err, loc=None, inflate=None, debug=False):
    """
    Function to do the ensemble kalman filter (EnKF) update with perturbed observations
    Originator: Zilu Meng

    Args:
        Xb: background ensemble estimates of state (Nx x Nens)
        obvalue: proxy value
        Ye: background ensemble estimate of the proxy (Nens x 1)
        ob_err: proxy error variance
        loc: localization vector (Nx x 1) [optional]
        inflate: scalar inflation factor [optional]
        debug: boolean flag to print debug info
    """
    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension # Nx x Nens

    # ensemble mean and variance of the background estimate of the proxy
    mye   = np.mean(Ye) # scalar
    varye = np.var(Ye,ddof=1) # scalar

    # lowercase ye has ensemble-mean removed
    ye = np.subtract(Ye, mye) # Nens 

    # innovation
    try:
        innov = obvalue - mye # scalar
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err) # scalar

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1) # Nx x 1

    # Option to inflate the covariances by a certain factor
    if inflate is not None:
        kcov = inflate * kcov

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)

    # Kalman gain
    kmat = np.divide(kcov, kdenom) # Nx x 1

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov) # Nx

    # update the ensemble members using the square-root approach
    # beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    beta = 1
    kmat = np.multiply(beta,kmat) # Nx x 1
    ye   = np.array(ye)[np.newaxis] # 1 x Nens
    # random perturbation 
    ye_random  =  np.random.normal(0,np.sqrt(ob_err),Nens)[np.newaxis]  # 1 x Nens
    ye = ye + ye_random # 1 x Nens
    kmat = np.array(kmat)[np.newaxis] # 1 x Nx
    Xap  = Xbp - np.dot(kmat.T, ye) # Nx x Nens

    # full state
    Xa = np.add(xam[:,None], Xap) # Nx x Nens

    # if masked array, making sure that fill_value = nan in the new array
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    # Return the full state
    return Xa


    


def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, debug=False):
    """ Function to do the ensemble square-root filter (EnSRF) update

    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator:
        G. J. Hakim, with code borrowed from L. Madaus Dept. Atmos. Sciences, Univ. of Washington

    Revisions:
        1 September 2017: changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) for an unbiased calculation of the variance. (G. Hakim - U. Washington)

    Args:
        Xb: background ensemble estimates of state (Nx x Nens)
        obvalue: proxy value
        Ye: background ensemble estimate of the proxy (Nens x 1)
        ob_err: proxy error variance
        loc: localization vector (Nx x 1) [optional]
        inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)

    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    # Return the full state
    return Xa



def mask_obs(OBS_pdb,mask_rate,mask_seed,logger=None):
    """
    mask the PBD for independent verification
    OBS_pdb: ProxyDatabase
    mask_rate: float, the rate of masking (0.2 meaning 20% of the proxies are masked)
    mask_seed: int, the seed for random mask
    logger: logger
    """
    if logger is None:
        logger = slim.get_logger()
    logger.info("Masking OBS with rate {}".format(mask_rate))
    obs_num = len(OBS_pdb.records)
    np.random.seed(mask_seed)
    mask_idx = np.random.choice([True, False], size=obs_num, p=[1- mask_rate, mask_rate])
    logger.info("Masked Ratio: {}".format(1 - mask_idx.sum()/obs_num))
    pdb_using = {}
    pdb_unused = {}
    for i, (pname, proxy) in enumerate(OBS_pdb.records.items()):
        if mask_idx[i]:
            pdb_using[pname] = proxy
        else:
            pdb_unused[pname] = proxy
    logger.info("used Proxy Number: {}".format(len(pdb_using)))
    logger.info("Masked Proxy Number: {}".format(len(pdb_unused)))
    return cfr.ProxyDatabase(pdb_using), cfr.ProxyDatabase(pdb_unused),mask_idx


def float2inttime(float_time):
    year = int(float_time)
    month = int((float_time - year) * 12) + 1
    return year, month


def get_index4proxy(lat,lon,lats=np.arange(-89,90,2),lons=np.arange(0,360,2),logger=None):
    lat_idx = np.abs(lats - lat).argmin()
    lon_idx = np.abs(lons - lon).argmin()
    if logger is not None:
        logger.info("Proxy Location: lat_idx: {}, lon_idx: {}, depature: {}".format(lat_idx,lon_idx, np.sqrt((lats[lat_idx] - lat)**2 + (lons[lon_idx] - lon)**2)))
    return lat_idx, lon_idx



def update_ensemble(ensemble_pool,eofs,LIM_config,pdb,current_time,logger=None,show_proxy=False,pseudo_label=False,update_func=enkf_update_array):
    """
    funcation for updating the ensemble
    ensemble_pool: 4, ens_num, pc_num
    eofs: eofs for transforming the pcs to the real space
    LIM_config: configuration for LIM
    pdb: ProxyDatabase
    current_time: float, the current time
    logger: logger
    show_proxy: bool, show the proxy information; default False
    pseudo_label: bool, use the pseudo proxy or real proxy; default False
    update_method: str, the method for updating the ensemble, 'EnSRF' or 'EnKF'
    """
    if logger is None: logger = slim.get_logger()
    # cycle through all proxies
    updated_proxy_num = 0
    for pid, proxy in pdb.records.items():
        year, month = float2inttime(current_time)
        start_yr = year - 1/2
        end_yr = year + 1/2 
        seasonality = proxy.psm.calib_details['seasonality']
        if seasonality[-1] == (month + 1) and year in proxy.time: # Feb for [-12,1,2], so +1; and year in proxy.time
            if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
            num4average = len(seasonality) // 3 # [3,4,5] for 1, [3,4,5,6,7,8] for 2
            selected_ens = ensemble_pool[-num4average:] # selected_num ,ens_num, pc_num
            averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
            lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
            obs_variable = proxy.psm.calib_details['df'].columns[1]
            pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
            pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
            if pseudo_label:
                time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
            else:
                time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)

            nobs = np.sum(time_mask)
            if pseudo_label:
                real_proxy = proxy.pseudo.value[time_mask].mean()
            else:
                real_proxy = proxy.value[time_mask].mean()

            if pseudo_label:
                ob_err = proxy.pseudo.R
            else:
                ob_err = proxy.R 
            if nobs >= 2: 
                logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
                continue
            selected_num,ens_num = selected_ens.shape[:-1]
            pc_num = selected_ens.shape[-1]
            Xb = selected_ens.swapaxes(1,2) # selected_num, pc_num, ens_num
            Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
            Xa = update_func(Xb,obvalue=real_proxy,Ye=pseudo_proxy,ob_err=ob_err) # selected_num * pc_num, ens_num
            Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
            ensemble_pool[-num4average:] = Xa
            updated_proxy_num += 1
    logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
    return ensemble_pool


def update_ensemble_just_end(ensemble_pool,eofs,LIM_config,pdb,current_time,logger=None,show_proxy=False,pseudo_label=False,update_func=enkf_update_array):
    """
    update ensemble just at the end of proxy seasonality
    For example, for proxy seasonality [3,4,5,6,7,8], only update at [6,7,8]
    """
    if logger is None: logger = slim.get_logger()
    # cycle through all proxies
    updated_proxy_num = 0
    for pid, proxy in pdb.records.items():
        year, month = float2inttime(current_time)
        start_yr = year - 1/2
        end_yr = year + 1/2 
        seasonality = proxy.psm.calib_details['seasonality']
        if seasonality[-1] == (month + 1) and year in proxy.time: # Feb for [-12,1,2], so +1; and year in proxy.time
            if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
            num4average = len(seasonality) // 3 # [3,4,5] for 1, [3,4,5,6,7,8] for 2
            selected_ens = ensemble_pool[-num4average:]
            averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
            lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
            obs_variable = proxy.psm.calib_details['df'].columns[1]
            pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
            pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
            if pseudo_label:
                time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
            else:
                time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)

            nobs = np.sum(time_mask)
            if pseudo_label:
                real_proxy = proxy.pseudo.value[time_mask].mean()
            else:
                real_proxy = proxy.value[time_mask].mean()

            if pseudo_label:
                ob_err = proxy.pseudo.R
            else:
                ob_err = proxy.R 
            if nobs >= 2: 
                logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
                continue
            selected_num,ens_num = selected_ens.shape[:-1]
            pc_num = selected_ens.shape[-1]
            Xb = selected_ens.swapaxes(1,2) # selected_num, pc_num, ens_num
            Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
            Xa = update_func(Xb,obvalue=real_proxy,Ye=pseudo_proxy,ob_err=ob_err) # selected_num * pc_num, ens_num
            Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
            # ensemble_pool[-num4aver
            # age:] = Xa
            # ================== only update the last month ==================
            ensemble_pool[-1:] = Xa[-1:] # only update the last month, 
            # ==============================================================
            updated_proxy_num += 1
    logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
    return ensemble_pool


def save_da_result(save_pool,save_time,main_path,logger=None):
    if logger is None: logger = slim.get_logger()
    if len(save_pool) == 0:
        logger.info("No data to save")
        return [],[]
    else:
        save_pool = np.array(save_pool)
        save_time = np.array(save_time)
        time_s = save_time[0]
        time_e = save_time[-1]
        year_s, month_s = float2inttime(time_s)
        year_e, month_e = float2inttime(time_e)
        save_path = main_path + "./data/{}-{}_{}-{}.nc".format(year_s,month_s,year_e,month_e)
        coords = {'time': save_time,'ens_num': np.arange(save_pool.shape[1]),'pc_num': np.arange(save_pool.shape[2])}
        dataarray = xr.DataArray(save_pool,coords=coords)
        dataset = xr.Dataset({'pcs': dataarray})
        dataset.to_netcdf(save_path)
        logger.info("Data saved at: {}, time: from {}-{} to {}-{}, data_length: {}".format(save_path,year_s,month_s,year_e,month_e,save_pool.shape[0]))
        save_number = save_pool.shape[0]
        return [],[],save_number


def update_ensemble_endofyear(ensemble_pool,eofs,LIM_config,pdb,current_time,logger=None,show_proxy=False,pseudo_label=False,update_func=enkf_update_array):
    """
    update ensemble at the end of year (9,10,11)
    ensemble_pool: 4, ens_num, pc_num
    """
    if logger is None: logger = slim.get_logger()
    updated_proxy_num = 0
    year, month = float2inttime(current_time)
    # update at 9 10 11 
    if month == 10: # SON
        for pid, proxy in pdb.records.items():
            start_yr = year - 1/2
            end_yr = year + 1/2 
            seasonality = proxy.psm.calib_details['seasonality']
            if year in proxy.time:
                if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
                num4average = len(seasonality) // 3
                start_month = abs(seasonality[0])
                end_month = abs(seasonality[-1])
                start_idx_dict = {12:0, 3:1, 6:2, 9:3}
                start_idx = start_idx_dict.get(start_month,None)
                if start_idx is None:
                    logger.info("Proxy {} has seasonality: {}, pass".format(pid, seasonality))
                    continue
                end_idx = start_idx + num4average
                selected_ens = ensemble_pool[start_idx:end_idx] # selected_num ,ens_num, pc_num
                averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
                lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
                obs_variable = proxy.psm.calib_details['df'].columns[1]
                pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
                pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
                if pseudo_label:
                    time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
                else:
                    time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)
                nobs = np.sum(time_mask)
                if pseudo_label:
                    real_proxy = proxy.pseudo.value[time_mask].mean()
                else:
                    real_proxy = proxy.value[time_mask].mean()  
                if pseudo_label:
                    ob_err = proxy.pseudo.R
                else:
                    ob_err = proxy.R
                if nobs >= 2:
                    logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
                    continue
                selected_num,ens_num = ensemble_pool.shape[:-1] # specific for end of year
                pc_num = ensemble_pool.shape[-1]
                Xb = ensemble_pool.swapaxes(1,2) # selected_num, pc_num, ens_num
                Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
                Xa = update_func(Xb,obvalue=real_proxy,Ye=pseudo_proxy,ob_err=ob_err) # selected_num * pc_num, ens_num
                Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
                ensemble_pool = Xa
                updated_proxy_num += 1
        logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
    else:
        logger.info(str(current_time) + " is not October, pass")
    return ensemble_pool


def update_ensemble_endofyear_error_matrix(ensemble_pool,eofs,LIM_config,pdb,current_time,logger=None,show_proxy=False,pseudo_label=False,error_matrix=None,update_func=enkf_update_array):
    """
    use the error matrix to update the ensemble
    first do eigen decomposition of the error matrix, then update the ensemble R = E * Rk * E^T
    Then yk = E^T * y
    Then use the yk series to update the ensemble
    """
    if logger is None: logger = slim.get_logger()
    updated_proxy_num = 0
    year, month = float2inttime(current_time)
    # update at 9 10 11
    if month == 10: # SON
        availabel_proxy = {}
        for pid, proxy in pdb.records.items(): # get the exsitent proxies
            if year in proxy.time:
                if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
                availabel_proxy[pid] = proxy
        if len(availabel_proxy) == 0:
            logger.info("No proxy available for updating")
        else:
            error_matrix_part = error_matrix[list(availabel_proxy.keys())].loc[list(availabel_proxy.keys())].to_numpy()
            # check the error matrix if it is asymentric or not
            max_save_error = np.max(np.abs(error_matrix_part - error_matrix_part.T))
            if max_save_error > 1e-5: raise ValueError("Error Matrix is not symmetric, please check")
            error_eigenvalues_or, error_eigenvectors_or = np.linalg.eigh(error_matrix_part)
            # filter the negative value
            positive_mask = (error_eigenvalues_or > 0)
            error_eigenvalues = error_eigenvalues_or[positive_mask]
            error_eigenvectors = error_eigenvectors_or[:,positive_mask]
            # error_eigenvalues, error_eigenvectors = error_eigenvalues[::-1],error_eigenvectors[:,::-1]
            # loop for new obs
            for new_proxy_idx in range(len(error_eigenvalues)):
                # loop for estimate the new obs from pcs
                eignen_vector_using = error_eigenvectors[:,new_proxy_idx]
                eignen_value_using = error_eigenvalues[new_proxy_idx]
                estimated_obses = []
                real_obses = []
                for pname, proxy in availabel_proxy.items():
                    start_yr = year - 1/2
                    end_yr = year + 1/2
                    # calculate the obs from pcs
                    seasonality = proxy.psm.calib_details['seasonality']
                    start_month = abs(seasonality[0])
                    end_month = abs(seasonality[-1])
                    start_idx_dict = {12:0, 3:1, 6:2, 9:3}
                    start_idx = start_idx_dict.get(start_month,None)
                    if start_idx is None:
                        logger.info("Proxy {} has seasonality: {}, pass".format(pid, seasonality))
                        continue
                    num4average = len(seasonality) // 3
                    end_idx = start_idx + num4average
                    selected_ens = ensemble_pool[start_idx:end_idx] # selected_num ,ens_num, pc_num
                    averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
                    lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
                    obs_variable = proxy.psm.calib_details['df'].columns[1]
                    pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
                    pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
                    if pseudo_label:
                        time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
                    else:
                        time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)
                    nobs = np.sum(time_mask)
                    if pseudo_label:
                        real_proxy = proxy.pseudo.value[time_mask].mean()
                    else:
                        real_proxy = proxy.value[time_mask].mean()
                    if nobs >= 2:
                        logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
                        continue
                    real_obses.append(real_proxy)
                    estimated_obses.append(pseudo_proxy) # ens_num
                # updtae the ensemble
                real_obses = np.array(real_obses) # len(availabel_proxy)
                estimated_obses = np.array(estimated_obses) # len(availabel_proxy), ens_num
                # logger.info([real_obses.shape,estimated_obses.shape])
                # transform the real_obses and estimated_obses to the new space
                yk = np.dot(eignen_vector_using,real_obses) # 1
                # print("yk",yk)
                estimated_yk = np.dot(eignen_vector_using,estimated_obses) # ens_num
                # print("eyk_m",estimated_yk.mean())
                # print("ensemle_pool_mean",ensemble_pool.shape,ensemble_pool.mean(),ensemble_pool.max(),ensemble_pool.min())
                selected_num,ens_num = ensemble_pool.shape[:-1] # specific for end of year
                pc_num = ensemble_pool.shape[-1]
                Xb = ensemble_pool.swapaxes(1,2) # selected_num, pc_num, ens_num
                Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
                # print('eignen_value_using:',eignen_value_using)
                Xa = update_func(Xb,obvalue=yk,Ye=estimated_yk,ob_err=eignen_value_using) # selected_num * pc_num, ens_num
                Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
                ensemble_pool = Xa
                # print("ensemle_pool_mean_after_update",ensemble_pool.shape,ensemble_pool.mean(),ensemble_pool.max(),ensemble_pool.min())

                updated_proxy_num += 1
            logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
    else:
        logger.info(str(current_time) + " is not October, pass")
    return ensemble_pool

                
    

def update_ensemble_noseasonal(ensemble_pool,eofs,LIM_config,pdb,current_time,logger=None,show_proxy=False,pseudo_label=False,update_func=enkf_update_array):
    """
    ensemble_pool: 4, ens_num, pc_num
    """
    if logger is None: logger = slim.get_logger()
    # cycle through all proxies
    updated_proxy_num = 0
    year, month = float2inttime(current_time)
    if month == 10:
        for pid, proxy in pdb.records.items():
            start_yr = year - 1/2
            end_yr = year + 1/2 
            seasonality = proxy.psm.calib_details['seasonality']
            if year in proxy.time:
                if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
                num4average = len(seasonality) // 3
                start_month = seasonality[0]
                end_month = seasonality[-1]
                if abs(start_month) == 12:
                    start_idx = 0
                elif start_month == 3:
                    start_idx = 1
                elif start_month == 6:
                    start_idx = 2
                elif start_month == 9:
                    start_idx = 3
                else: 
                    continue
                end_idx = start_idx + num4average
                selected_ens = ensemble_pool[start_idx:end_idx] # selected_num ,ens_num, pc_num
                averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
                lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
                obs_variable = proxy.psm.calib_details['df'].columns[1]
                pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
                pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
                if pseudo_label:
                    time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
                else:
                    time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)
                nobs = np.sum(time_mask)
                if pseudo_label:
                    real_proxy = proxy.pseudo.value[time_mask].mean()
                else:
                    real_proxy = proxy.value[time_mask].mean()
                if pseudo_label:
                    ob_err = proxy.pseudo.R
                else:
                    ob_err = proxy.R
                if nobs >= 2:
                    logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
                    continue
                selected_num,ens_num = selected_ens.shape[:-1]
                pc_num = selected_ens.shape[-1]
                Xb = selected_ens.swapaxes(1,2) # selected_num, pc_num, ens_num
                Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
                Xa = update_func(Xb,obvalue=real_proxy,Ye=pseudo_proxy,ob_err=ob_err) # selected_num * pc_num, ens_num
                Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
                ensemble_pool[start_idx:end_idx] = Xa
                updated_proxy_num += 1
    else:
        logger.info(str(current_time) + " is not October, pass")
    logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
    return ensemble_pool



            
        #     if seasonality[-1] == (month + 1) and year in proxy.time: # Feb for [-12,1,2], so +1; and year in proxy.time
        #         if show_proxy: logger.info("Update Ensemble for Proxy: {}".format(pid))
        #         num4average = len(seasonality) // 3 # [3,4,5] for 1, [3,4,5,6,7,8] for 2
        #         selected_ens = ensemble_pool[-num4average:] # selected_num ,ens_num, pc_num
        #         averaged_ens = np.mean(selected_ens,axis=0) # ens_num, pc_num
        #         lat_idx, lon_idx = get_index4proxy(proxy.lat,proxy.lon)
        #         obs_variable = proxy.psm.calib_details['df'].columns[1]
        #         pseudo_proxy = lim_utils.decoder_specific_locations(pcs=averaged_ens,eofs=eofs,infos=LIM_config['vars_info'],var_obs=obs_variable,locations=[lat_idx,lon_idx]) # ens_num
        #         pseudo_proxy = proxy.psm.model.predict({obs_variable: pseudo_proxy}).values
        #         if pseudo_label:
        #             time_mask =( proxy.pseudo.time >= start_yr) & (proxy.pseudo.time <= end_yr)
        #         else:
        #             time_mask =( proxy.time >= start_yr) & (proxy.time <= end_yr)

        #         nobs = np.sum(time_mask)
        #         if pseudo_label:
        #             real_proxy = proxy.pseudo.value[time_mask].mean()
        #         else:
        #             real_proxy = proxy.value[time_mask].mean()

        #         if pseudo_label:
        #             ob_err = proxy.pseudo.R
        #         else:
        #             ob_err = proxy.R 
        #         if nobs >= 2: 
        #             logger.info("Proxy {} has {} observations, passed".format(pid,nobs))
        #             continue
        #         selected_num,ens_num = selected_ens.shape[:-1]
        #         pc_num = selected_ens.shape[-1]
        #         Xb = selected_ens.swapaxes(1,2) # selected_num, pc_num, ens_num
        #         Xb = Xb.reshape(selected_num * pc_num,ens_num) # selected_num * pc_num, ens_num
        #         Xa = enkf_update_array(Xb,obvalue=real_proxy,Ye=pseudo_proxy,ob_err=ob_err) # selected_num * pc_num, ens_num
        #         Xa = Xa.reshape(selected_num,pc_num,ens_num).swapaxes(1,2) # selected_num, ens_num, pc_num
        #         ensemble_pool[-num4average:] = Xa
        #         updated_proxy_num += 1
        # logger.info("Updated Proxy Number: {}".format(updated_proxy_num))
        # return ensemble_pool



            
        
            
        


            
    
    