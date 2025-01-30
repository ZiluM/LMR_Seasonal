import cftime
import numpy as np
import pandas as pd
import xarray as xr
import logging
import sys

def get_logger(save_path=None,mode=None):
    if mode is None:
        mode = 'w'
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if save_path is not None:
        handler = logging.FileHandler(save_path, mode=mode, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        pass
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def time_range(time_range):

    # if not hasattr(self, "time_range"):
    #     self.time_range = time_range_dict[self.exp_name]
    time_begin, time_end = time_range.split("-")
    # time_begin = cftime.DatetimeGregorian(time_str[:4], 1, 1)
    time_begin, time_end = map(lambda time_str: np.datetime64(cftime.DatetimeGregorian(int(time_str[:4]), int(time_str[4:]),1).isoformat(),"M"), [time_begin, time_end])
    time_range = np.arange(time_begin, time_end + np.timedelta64(1, 'M'), np.timedelta64(1, 'M'),)
    time_range = [cftime.datetime.strptime(tm,'%Y-%m', calendar='noleap') for tm in time_range.astype('str')] 
    return time_range

# def time_array1()

def single_time(year,month):
    time = np.datetime64(cftime.DatetimeGregorian(int(year), int(month),1).isoformat(),"M")
    time = cftime.datetime.strptime(time.astype("str"),'%Y-%m', calendar='noleap')
    return time

def multimes(year1,month1,tlength):
    delta_year = tlength // 12
    delta_month = tlength % 12
    if delta_month + month1 > 12:
        delta_year += 1
        delta_month = delta_month + month1 - 12
    year2 = year1 + delta_year
    month2 = month1 + delta_month
    res = time_range(str(year1)+str(month1)+"-"+str(year2)+str(month2))[:-1] # remove the last month to keep the same length

    # time_range = []
    # current = single_time(year1,month1)
    # time_range.append(current)
    # for date in range(tlength):
    #     current = current + np.timedelta64(1, 'M')
    #     time_range.append(current)
    # time_range = [cftime.datetime.strptime(tm,'%Y-%m', calendar='noleap') for tm in time_range.astype('str')] 
    return res
        # time_range.append( single_time(year1,month1) + np.timedelta64(1, 'M'))

    

# def mul_times(years,months):
#     time_array = 



def field_corr(field1, field2):
    """
    field1: time, nspace
    field2: time, nspace
    """
    field1a = field1 - field1.mean(axis=0)
    field2a = field2 - field2.mean(axis=0)
    covar = np.einsum("ij...,ij...->j...", field1a, field2a) / (field1a.shape[0] - 1)  # covar:nspace
    corr = covar / np.std(field1a, axis=0) / np.std(field2a, axis=0)  # corr: nspace
    return corr.real

def field_ce(field1, field2):
    """
    calculate the coefficient of efficiency
    field1: time, nspace
    field2: time, nspace: validation
    """
    field2m = np.nanmean(field2,axis=0)
    field2a = field2 - field2m
    field2_var = np.einsum("ij...,ij...->j...", field2a, field2a) 
    field1a = field1 - field2m
    field1_var = np.einsum("ij...,ij...->j...", field1a, field1a)
    ce = 1 - field1_var / field2_var
    return ce.real

def field_rmse(field1, field2):
    """
    calculate the root mean square error
    field1: time, nspace
    field2: time, nspace: validation
    """
    rmse = np.sqrt(np.nanmean((field1 - field2)**2,axis=0))
    return rmse.real
    
    