# coding=utf-8
"""
Main Cyclo-Stationary Linear Inverse Model classes and methods.

Zilu Meng, 2023
"""

import numpy as np
from numpy.linalg import pinv, eigvals, eig, eigh
import pickle
import sys
import logging
import util as utils



class CSLIM(object):

    def __init__(self, tau0_data, tau1_data=None, cycle_ind=None, fit_noise=False, cycle_labels=np.arange(1, 13),logger=None):
        if logger is None:
            logger = utils.get_logger()

        self.logger = logger
        self.cycle_labels = cycle_labels
        
        if tau0_data.ndim != 2:
            logger.error(('LIM calibration data is not 2D '
                          '(Contained ndim={:d}').format(tau0_data.ndim))
            raise ValueError('Input LIM calibration data is not 2D')
        if tau1_data is None:
            tau1_data = tau0_data[1:]
            tau0_data = tau0_data[:-1]
            self.get_tau = False 
        else:
            self.get_tau = True
        self.tau0_data = tau0_data
        self.tau1_data = tau1_data
        cycle_ind = np.array(cycle_ind)
        # if tau1_data is None:
        #     tau1_data = tau0_data[1:]
        #     tau0_data = tau0_data[:-1]
        C0_dict = {}
        C1_dict = {}
        G_dict = {}
        # L_dict = {}
        Geigs_dict = {}
        Leigs_dict = {}
        for month in cycle_labels:
            if self.get_tau:
                month_label = np.where(cycle_ind == month)[0]
            else:
                month_label = np.where(cycle_ind[:-1] == month)[0]
            tau0_0 = tau0_data[month_label]
            tau1_0 = tau1_data[month_label]
            logger.info('Calibrating month {:d}, number is {:d}'.format(month, tau0_0.shape[0]))
            C_idx0 = tau0_0.T @ tau0_0 / (tau0_0.shape[0] - 1)
            C_idx1 = tau1_0.T @ tau0_0 / (tau0_0.shape[0] - 1)
            C0_dict[month] = C_idx0
            C1_dict[month] = C_idx1
            Gidx = np.dot(C_idx1, pinv(C_idx0))
            G_dict[month] = Gidx
            Geigs = eigvals(Gidx)
            Leigs = (1. / 1) * np.log(Geigs)
            if np.any(Leigs.real >= 0):
                logger.info('Positive eigenvalues detected in forecast matrix G., month: {:}'.format(month))
                # raise ValueError(
                # 'Positive eigenvalues detected in forecast matrix L.')
            
            Geigs_dict[month] = Geigs
            Leigs_dict[month] = Leigs
        self.C0_dict = C0_dict
        self.C1_dict = C1_dict
        self.G_dict = G_dict
        self.Geigs_dict = Geigs_dict
        self.Leigs_dict = Leigs_dict
        G_mul = np.eye(tau0_data.shape[1])
        for month in cycle_labels:
            G_mul = G_mul @ G_dict[month]
        G_mul_eigs = eigvals(G_mul)
        L_mul_eigs = np.log(G_mul_eigs)
        self.G_mul = G_mul
        self.G_mul_eigs = G_mul_eigs
        self.L_mul_eigs = L_mul_eigs
        if np.any(L_mul_eigs.real >= 1):
            logger.info('Warming !! Positive eigenvalues detected in forecast matrix L. ============================')
            logger.info(L_mul_eigs.real)
            
        if fit_noise:
            self.cal_noise()

    def forecast(self, data0, month0, length,sep=True):
        """
        data0: (ensemble number, pc number)
        month0: initial months
        length: length of forecast (month)
        sep: if True, month0 can be a list of months, and length should be a list of length
        """
        if sep is False:
            ltime = month0
            tlength = 0
            res_array = np.zeros((length, *data0.shape))
            while tlength < length:
                pre_data = data0 @ self.G_dict[ltime]
                ltime += 1
                ltime = ltime % self.cycle_labels.max() if ltime != self.cycle_labels.max() else self.cycle_labels.max()
                res_array[tlength] = pre_data.real
                data0 = pre_data
                tlength += 1
        if sep is True: 
            res_array = np.zeros((length, *data0.shape))
            for ltime in self.cycle_labels:
                labels = (month0 == ltime)
                pre_init = data0[labels] # time = 0 data for forecast
                if labels.sum() > 0:
                    tlength = 0
                    current_data = pre_init
                    while tlength < length:
                        G_label = ltime + tlength if ltime + tlength <= self.cycle_labels.max() else ltime + tlength - self.cycle_labels.max()
                        # pre_data = current_data @ self.G_dict[G_label]
                        pre_data = np.dot(self.G_dict[G_label], current_data.T).T # change the bug
                        res_array[tlength, labels] = pre_data.real
                        # data0[labels] = pre_data
                        tlength += 1
                        current_data = pre_data
                    
        return res_array
    
    def update_cov(self,Pa,month):
        """
        Pa:(pc_num,pc_num)
        month: local month
        """
        M = self.G_dict[month]
        determis = M @ Pa @ M.T
        # Q = self.Q_dict[month]
        # N_dt = self.C0_dict[month] - M @ self.C0_dict[month] @ M.T
        month_p1 = month + 1 if month != self.cycle_labels.max() else 1
        N_dt = -self.G_dict[month] @ self.C0_dict[month] @ self.G_dict[month].T + self.C0_dict[month_p1]
        return determis + N_dt
        


    def cal_noise(self, max_neg_evals=10e5):
        self.L_dict = {}
        self.Q_dict = {}
        self.q_evals_dict = {}
        self.q_evacts_dict = {}
        self.num_neg_dict = {}
        self.scale_factor_dict = {}
        self.Leigs_dict = {}
        for month in self.cycle_labels:
            G_eval, G_evects = eig(self.G_dict[month])
            L_evals = (1 / 1) * np.log(G_eval)
            L = G_evects @ np.diag(L_evals) @ pinv(G_evects)
            L = np.matrix(L)
            # L = np.log(self.G_dict[month])
            # plt.figure()
            # plt.imshow((np.exp(L) - self.G_dict[month]).real)
            # plt.colorbar()
            # plt.savefig(f"L_G{month}.png")
            # plt.show()
            idm1 = month - 1 if month != 1 else self.cycle_labels[-1]
            idp1 = month + 1 if month != self.cycle_labels.max() else 1
            Qidx = (self.C0_dict[idp1] - self.C0_dict[idm1]) / 2 - \
            (L @ self.C0_dict[month] + self.C0_dict[month] @ L.H)
            
            q_evals, q_evects = eigh(Qidx)
            is_adj = abs(Qidx - Qidx.H)
            tol = 1e-5
            if np.any(abs(is_adj) > tol):
                print(abs(is_adj).max())
                self.logger.info('Determined Q is not Hermetian (complex '
                                 'conjugate transpose is equivalent.)')
            sort_idx = q_evals.argsort()
            q_evals = q_evals[sort_idx][::-1]
            q_evects = q_evects[:, sort_idx][:, ::-1]
            num_neg = (q_evals < 0).sum()
            self.logger.info('Found {:} negative eigenvalues in the noise '.format(num_neg))
            if num_neg > 0:
                num_left = len(q_evals) - num_neg
                # if num_neg > max_neg_evals:
                self.logger.debug('Found {:} modes with negative eigenvalues in'
                                ' the noise covariance term, Q.'.format(num_neg))
                # self.logger.info('More than {:} negative eigenvalues of Q '
                                    # 'detected.  Consider further dimensional '
                                    # 'reduction.'.format(max_neg_evals))
            
                self.logger.info('Removing negative eigenvalues and rescaling {:} '
                            'remaining eigenvalues of Q.'.format(num_left))
                pos_q_evals = q_evals[q_evals > 0]
                scale_factor = q_evals.sum() / pos_q_evals.sum()
                self.logger.info('Q eigenvalue rescaling: {:1.2f}'.format(scale_factor))

                q_evals = q_evals[:-num_neg] * scale_factor
                # print(q_evals.shape)
                q_evects = q_evects[:, :-num_neg]
            else:
                scale_factor = None
            self.Q_dict[month] = np.array(Qidx)
            self.L_dict[month] = np.array(L)
            self.Leigs_dict[month] = L_evals
            self.q_evals_dict[month] = q_evals
            self.q_evacts_dict[month] = np.array(q_evects)
            self.num_neg_dict[month] = num_neg
            self.scale_factor_dict[month] = scale_factor
            # print(L.shape, q_evals.shape, q_evects.shape)

            # Leigs = (1. / ) * np.log(Geigs)
    def two_step(self,L, x, Q_evec, Q_eval, noise, tdelta):
        y = L @ x * tdelta + Q_evec @ (np.sqrt(Q_eval * tdelta) * noise) + x
        x = (y + x) / 2
        return x # x

    def noise_integration(self, data0, month0, length,dt=1/720, seed=None):
        """
        data0: (ensemble number, pc number)
        month0: initial month
        length: length of simulation (month)
        dt: time step (unit: month)
        seed: random seed
        """
        if seed is not None:
            np.random.seed(seed)
        simulation_time = 0
        all_simulation_step = length * int(1 / dt)
        simulation_step = 0
        save_step = int(1 / dt)
        time_label = month0
        res_array = np.zeros((length+1, *data0.shape)) # (time, ensemble, pc)
        # print("============Integrating Start============")
        # print(f"simulation step: {simulation_step}, time label: {time_label}, save step: {simulation_step // save_step}")
        res_array[0] = data0 # initial data
        ens_num = data0.shape[0]  # ensemble size
        state_1 = data0.T # pc_num, ens_num
        state_mid = state_1 # pc_num, ens_num
        # pass
        # while 
        while simulation_step < all_simulation_step + 1:
            Q_evec = self.q_evacts_dict[time_label]
            Q_eval = self.q_evals_dict[time_label][...,np.newaxis]
            L = self.L_dict[time_label]
            num_evals = Q_eval.shape[0]
            # start two step method ====================================
            random_1 = np.random.normal(size=(num_evals, ens_num))
            state_mid = self.two_step(L, state_1, Q_evec, Q_eval, random_1, tdelta=dt)
            # state_2 = state_mid
            # ========================================================
            random_2 = np.random.normal(size=(num_evals, ens_num))
            state_2  = self.two_step(L, state_mid, Q_evec, Q_eval, random_2, tdelta=dt)
            # end two step method ====================================
            simulation_step += 1
            if simulation_step % save_step == 0:
                res_array[simulation_step // save_step] = state_2.T.real
                time_label += 1
                time_label = time_label % self.cycle_labels.max() if time_label != self.cycle_labels.max() else self.cycle_labels.max()
                # print(f"simulation step: {simulation_step}, time label: {time_label}, save step: {simulation_step // save_step}")
            state_1 = state_2
        return res_array
    


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save_precalib(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self, f)




    # def noise_intergral(self, data0, month0, length,dt=1/30/4, seed=10):
    #     """
    #     data0: initial data
    #     month0: initial month
    #     length: length of forecast (month)
    #     dt: time step (month)
    #     seed: random seed

    #     """
    #     if seed is not None:
    #         np.random.seed(seed)
    #     tlength = 0
    #     all_length = length * int(1 / dt)
    #     ltime = month0 # start month
    #     res_array = np.zeros((length, *data0.shape))
    #     # num_evals = self.q_evals_dict[].shape[0]
    #     nens = data0.shape[0]  # ensemble size
    #     state_1 = data0.T  # essamble,
    #     state_mid = state_1
    #     save_step = int(1 / dt)
    #     while tlength < all_length:
    #         Q_evec = self.q_evacts_dict[ltime]
    #         Q_eval = self.q_evals_dict[ltime][...,np.newaxis]
    #         L = self.L_dict[ltime]
    #         # tdelta = 1

    #         # # deterministic =  @ data0
    #         # deterministic = (L @ state_1) * tdelta
    #         # # random = np.random.normal(size=(num_evals, nens))
    #         num_evals = Q_eval.shape[0]
    #         random_1 = np.random.normal(size=(num_evals, nens))
    #         # stochastic = Q_evec @ (np.sqrt(Q_eval * tdelta) * random)
    #         # state_2 = state_1 + deterministic + stochastic
    #         # state_mid = (state_1 + state_2) / 2
    #         # state_1 = state_2
    #         state_mid = self.two_step(L, state_1, Q_evec, Q_eval, random_1, tdelta=dt/2)
    #         random_2 = np.random.normal(size=(num_evals, nens))

    #         state_2  = self.two_step(L, state_mid, Q_evec, Q_eval, random_2, tdelta=dt/2)
    #         state_1 = state_2


    #         # stochastic = Q_evec @ (np.sqrt(Q_eval * tdelta) * random)
    #         # res_array[tlength] = pre_data
    #         # data0 = pre_data
    #         if tlength % save_step == 0:
    #             res_array[tlength // save_step] = state_2.T.real
    #             ltime += 1
    #             ltime = ltime % self.cycle_labels.max()
    #         tlength += 1
    #     return res_array


if __name__ == "__main__":
    import sacpy as scp
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    # import matplotlib.pyplot as plt
    # from EOF import EOF
    sst = xr.open_dataset("data/obs/HadISST_sst_180x90.nc")['sst'].loc[:, -30:30, 120:280]
    # ssta = scp.get_anom(sst)
    # ssta = xr.where(np.abs(ssta) > 40, np.NAN, ssta)
    # eof = EOF(ssta)
    # eof.solve(method="dask_svd", chunks=(300), dim_min=20)
    # # print(eof.get_varperc())
    # pc = eof.get_pc(npt=20)
    pc = np.load("./pckk.npy")
    # np.save("pckk.npy", pc)
    cslim = CSLIM(pc.T, cycle_ind=sst.time.dt.month)
    cslim.cal_noise()
    # res1 = cslim.forecast(pc.T[[0]], month0=1, length=10)
    res2 = cslim.noise_intergral(pc.T[[0]*13], month0=1, length=100)
    print(res2.shape)
    print(res2[0])
    pltt = -1
    # plt.plot(res1[pltt,0, :],'-o',label="forecast_1")
    # plt.plot(res2[pltt, :, :].mean(axis=0),'-o',label="forecast_2")
    # # plt.plot(pc[:, pltt + 1],label="truth")
    # plt.legend()
    # plt.savefig("kk.png")
    # plt.show()
