import numpy as np
import scipy.stats as sts
import xarray as xr
# from .LinReg import LinReg
import time
from time import gmtime, strftime
import pickle
import logging
# import sacpy
from .LIM import LIM
import pickle as cpk

logger = logging.getLogger(__name__)

try:
    import dask.array as dsa
except:
    pass

class SNLIM:
    """
    Seasonal Normalized Linear Inverse Model (SNLIM) class
    """
    def __init__(self,tau0_data,months,nMonths=np.arange(1,5)) -> None:
        """
        tau0_data: tLength x nPCs
        months: tLength
        nMonths: months label; For example, nMonths=[1,2,3,4] for DJF, MAM, JJA, SON; nMonths=[1,2,3,4,5,6,7,8,9,10,11,12] for all months; Default is [1,2,3,4]
        """
        tau0_data, seasonStds = self.normalize_season(tau0_data, months, nMonths)
        Lim = LIM(tau0_data,fit_noise=True)
        self.Lim = Lim
        self.seasonStds = seasonStds
        self.nMonths = nMonths

        # model parameters
        self.L, self.Q_evals, self.Q_evects, self.num_neg_Q, self.neg_Q_rescale_factor = Lim.L, Lim.Q_evals, Lim.Q_evects, Lim.num_neg_Q, Lim.neg_Q_rescale_factor
        self.L_eigs = Lim.L_eigs
        

    def normalize_season(self, data, months, nMonths=np.arange(1,5)):
        """
        Normalize the data by the standard deviation of the seasonal cycle.
        data: tLength x nPCs
        months: tLength
        nMonths: months label; For example, nMonths=[1,2,3,4] for DJF, MAM, JJA, SON; nMonths=[1,2,3,4,5,6,7,8,9,10,11,12] for all months; Default is [1,2,3,4]
        """
        tLength = data.shape[0]
        if tLength != months.shape[0]:
            raise ValueError("The length of the data and the months must be the same. rather than {} and {}".format(tLength, months.shape[0]))
        seasonStds = {} # month: seasonStd (nPCs,)
        for month in nMonths:
            seasonStd = data[months==month].std(axis=0) # shape: nPCs
            seasonStds[month] = seasonStd
            data[months==month] = data[months==month] / seasonStd
        
        return data, seasonStds
    
    def forecast(self, tau0_data, month0,fcast_leads):
        """
        forecast the data for the next fcast_leads months 
        ```
        X_{t+1} = G * X_{t} 
        ```
        where `G` is the linear operator from LIM
        tau0_data: tLength x nPCs
        month0: the month of the last data point; (tLength, )
        fcast_leads: number of months to forecast
        """
        Lim = self.Lim
        seasonStds = self.seasonStds
        nMonths = self.nMonths
        tLength = tau0_data.shape[0]
        if tau0_data.shape[0] != month0.shape[0]:
            raise ValueError("The length of the data and the months must be the same, not {} and {}".format(tau0_data.shape[0], month0.shape[0]))
        # normalize the data
        tau0_data_norm = tau0_data.copy()
        for month in nMonths:
            tau0_data_norm[month0==month] = tau0_data[month0==month] / seasonStds[month]
        # forecast the data
        forecastNorm = Lim.forecast(tau0_data_norm, np.arange(1,fcast_leads+1)) # shape: fcast_leads x tLength x nPCs

        # denormalize the forecast
        # calculate the month of the forecast
        forecastMonth = np.zeros([fcast_leads, tLength], dtype=int)
        for i in range(fcast_leads):
            forecastMonth[i] = month0 + i + 1
        forecastMonth = forecastMonth %  nMonths.max()
        forecastMonth[forecastMonth == 0] = nMonths.max()
        forecast = forecastNorm.copy()
        print(forecastMonth.shape, forecastNorm.shape, forecast.shape, seasonStds.keys())
        for month in nMonths:
            forecast[forecastMonth==month] = forecastNorm[forecastMonth==month] * seasonStds[month]
        return forecast
        
        
        

    def noise_integration(self, tau0_data, length, month0, seed=None):
        """
        Integrate:
        ```
        X_{t+1} = G * X_{t} + N_{t}
        ```
        where `G` is the linear operator from LIM and `N_{t}` is the noise from LIM
        tau0_data: ensemble number x nPCs
        length: number of months to integrate (at lease 2, 1+1); if you want to integrate 1 month, length = 2; 
        month0: the month of the last data point; (1,)
        seed: random seed for the noise
        """
        Lim = self.Lim
        seasonStds = self.seasonStds
        nMonths = self.nMonths
        ensNum = tau0_data.shape[0]
        length_out_arr = np.zeros([length, ensNum, tau0_data.shape[1]])
        # normalize the data
        tau0_data_norm = tau0_data.copy()
        tau0_data_norm = tau0_data_norm / seasonStds[month0]
        # integrate the data
        inted = Lim.noise_integration(tau0_data_norm, length, length_out_arr=length_out_arr, seed=seed)
        # denormalize the data
        outputMonth = month0 + np.arange(length)
        outputMonth = outputMonth % nMonths.max()
        outputMonth[outputMonth == 0] = nMonths.max()
        for month in nMonths:
            inted[outputMonth==month] = inted[outputMonth==month] * seasonStds[month]
        return inted

    def save_precalib(self, filename):

        with open(filename, 'wb') as f:
            cpk.dump(self, f)

        print('Saved pre-calibrated LIM to {}'.format(filename))



