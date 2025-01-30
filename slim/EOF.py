"""
EOF analysis of data

Author: Zilu Meng
"""

import numpy as np
import scipy.stats as sts
import xarray as xr
# from .LinReg import LinReg
import time
from time import gmtime, strftime
import pickle
# import sacpy


try:
    import dask.array as dsa
except:
    pass

EPS = 1e-5


class EOF:
    """ EOF analysis of data
    """
    name = "EOF"

    def __init__(self, data: np.ndarray, weights=None,channel_std=None):
        """ initiation of EOF
        Args:
            data (np.ndarray): shape (time, * space grid number)
            weights : shape (* space grid number , 
                        or can be broadcast to space grid number)
        """
        # original data
        if isinstance(data, xr.DataArray):
            data = np.array(data)
        if weights is None:
            self.data = np.copy(data)
        else:
            self.data = np.copy(data) * weights
        # original data shape
        self.weights = weights
        self.origin_shape = data.shape
        # time length
        self.tLen = data.shape[0]
        self.channel_std = channel_std
        # reshape (time, space)
        self.rsp_data = data.reshape(self.tLen, -1)
        self.pc = None
        self.got_pc_num = 0
    
    def save(self, path):
        del self.data
        del self.rsp_data
        del self.data_nN
        with open(path, "wb") as f:
            pickle.dump(self, f)


    def _mask_nan(self):
        """ mask Nan or drop out the Nan
        """
        # get data0
        data0 = self.rsp_data
        # get not Nan flag
        flag = np.isnan(data0).sum(axis=0) == 0
        # save to self.flag
        self.flag = flag
        # get data without Nan
        data_nN = data0[:, flag]  # (time, space_noNan)
        # save
        self.data_nN = data_nN

    # def _mask_extra_data(self,data):
    #     flag = np.isnan(data0).sum(axis=0) == 0

    def solve(self, method="eig",st=False,chunks=None,dim_min=None):
        """ solve the EOF
        """
        if method not in ['eig', 'svd','dask_svd']:
            raise ValueError(f"method must be 'eig' or 'svd', not {method}")
        # mask data
        self._mask_nan()
        # solve maksed data
        data_nN = self.data_nN
        # get (time,spcae_noNan)
        dim0_len, dim1_len = data_nN.shape
        # print("Start EOF")
        # ================================= EOF process by SVD===============================
        if st is True:
            print("=====EOF Start at {}======".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        if method == "svd":
            Xor = 1 / np.sqrt(dim0_len - 1) * data_nN
            U, Sigma, VT = np.linalg.svd(Xor)
            e_vector = VT
            eign = Sigma
            if dim_min is None:
                dim_min = np.min([dim0_len, dim1_len])
        # ================================= EOF process end===============================
        elif method == "eig":
            if dim0_len > dim1_len:  # time > space
                # print(1)
                if dim_min is None:
                    dim_min = dim1_len
                # get coviance (space_noNan,space_noNan)
                cov = data_nN.T @ data_nN
                # get eigenvalues and right eigenvectors
                eign, e_vector = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
                # trans
                e_vector = e_vector.T  # [i]&[i,:]

            else:  # space > time
                # print(2)
                # get coviance
                if dim_min is None:
                    dim_min = dim0_len
                # get cov, (time,time)
                cov = data_nN @ data_nN.T
                # get eigenvalues and right eigenvectors
                eign, e_vector_s = np.linalg.eig(cov)  # (dim_min) ; (dim_min , dim_min) [i]&[:,i]
                # trans
                e_vector = (data_nN.T @ e_vector_s / np.sqrt(np.abs(eign))).T[:dim_min]
        elif method == "dask_svd":
            if chunks is None:
                raise ValueError(f"chunks must can't be None")
            Xor = 1 / np.sqrt(dim0_len - 1) * data_nN
            Xor = dsa.from_array(Xor, chunks=chunks)
            if dim_min is None: 
                dim_min = np.min([dim0_len, dim1_len])
            U, Sigma, V = dsa.linalg.svd_compressed(Xor, k=dim_min)
            U.compute()
            Sigma.compute()
            V.compute()
            e_vector = np.array(V)
            eign = np.array((Sigma**2).compute())
        if st is True:
            print("=====EOF End at  {}======".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        # save
        # print("EOF End")
        self.e_vector = e_vector
        self.eign = eign
        self.dim_min = dim_min
        # get patterns
        patterns = np.zeros((dim_min, *self.rsp_data.shape[1:]))
        patterns[:, self.flag] = e_vector[:dim_min]
        # refill Nan
        patterns[:, np.logical_not(self.flag)] = np.NAN
        # patterns = patterns.reshape((dim_min, *self.origin_shape[1:]))
        # save
        self.patterns_num = patterns.shape[0]
        self.patterns = patterns

    def get_eign(self):
        """ get eign of each pattern
        """
        return self.eign

    def get_varperc(self, npt=None):
        """ return variance percential

        Args:
            npt (int, optional): n patterns to get. Defaults to None.

        Returns:
            variace percentile (np.ndarray): variance percentile (npt,)
        """
        if npt is None:
            npt = self.dim_min
        var_perc = self.eign[:npt] / np.sum(self.data_nN.var(axis=0))
        self.data_var = np.sum(self.data_nN.var(axis=0))
        self.var_perc = var_perc
        return var_perc

    def get_pc(self, npt=None, scaling="std"):
        """ get pc of eof analysis

        Args:
            scaling (str, optional): scale method. None, 'std','DSE' and 'MSE'. Defaults to "std".

        Returns:
            pc_re : pc of eof (pattern_num, time)
        """

        if npt is None:
            npt = self.dim_min
        pc = self.e_vector[:npt] @ self.data_nN.T  # (pattern_num, time)
        # self.pc = pc
        self.pc_std = pc[:npt].std(axis=1)[..., np.newaxis]
        self.pc_scaling = scaling
        if scaling == "std":
            pc_re = pc[:npt] / self.pc_std
        elif scaling == "DSE":
            pc_re = pc[:npt] / np.sqrt(self.eign[:npt, ..., np.newaxis])
        elif scaling == "MSE":
            pc_re = pc[:npt] * np.sqrt(self.eign[:npt, ..., np.newaxis])
        elif scaling is None:
            pc_re = pc
        else:
            raise ValueError(f"invalid PC scaling option: '{scaling}', Must be None, 'std','DSE' and 'MSE' ")
        self.pc = pc_re
        self.got_pc_num = npt
        return pc_re

    def get_pt(self, npt=None, scaling="mstd"):
        """ get spatial patterns of EOF analysis

        Args:
            scaling (str, optional): sacling method. None, 'std','DSE' and 'MSE'. Defaults to "mstd".

        Returns:
            patterns : (pattern_num, * space grid number)
        """
        if npt is None:
            npt = self.dim_min
        if self.pc is None or self.got_pc_num < npt:
            self.get_pc(npt=npt)
        if scaling == "mstd":
            # print(self.patterns[:npt].shape)
            # print(self.pc_std[:npt, ..., np.newaxis].shape)
            patterns = self.patterns[:npt] * self.pc_std[:npt]
        elif scaling == "MSE":
            patterns = self.patterns[:npt] * self.eign[:npt, ..., np.newaxis]
        elif scaling == "DSE":
            patterns = self.patterns[:npt] / self.eign[:npt, ..., np.newaxis]
        else:
            patterns = self.patterns[:npt]
        # reshape to original shape (pattern_num,*space)
        patterns = patterns.reshape((npt, *self.origin_shape[1:]))
        self.pt = patterns
        return patterns

    # def correlation_map(self, npt):
    #     """ Get correlation map

    #     Args:
    #         npt 

    #     Returns:
    #         correlation map (npt, ...)
    #     """
    #     pcs = self.get_pc(npt=npt)
    #     corr_map_ls = []
    #     for i in range(npt):
    #         pc = pcs[i]
    #         Lin_map = LinReg(pc, self.data).corr
    #         corr_map_ls.append(Lin_map[np.newaxis, ...])
    #     corr_maps = np.concatenate(corr_map_ls, axis=0)
    #     return corr_maps

    def projection(self, proj_field: np.ndarray, npt=None, scaling="std"):
        """ project new field to EOF spatial pattern

        Args:
            proj_field (np.ndarray): shape (time, *space grid number)
            scaling (str, optional): _description_. Defaults to "std".

        Returns:
            pc_proj : _description_
        """
        if npt is None:
            npt = self.dim_min
        if self.weights is not None:
            proj_field = proj_field * self.weights
        if self.channel_std is not None:
            proj_field  = proj_field  / self.channel_std 
        proj_field_noNan = proj_field.reshape(proj_field.shape[0], -1)[:, self.flag]
        pc_proj = self.e_vector @ proj_field_noNan.T
        if self.pc is None or self.got_pc_num < npt:
            self.get_pc(npt)
        if scaling == "std":
            pc_proj = pc_proj / self.pc_std
        return pc_proj

    # def decoder(self):
    #     pass

    def load_pt(self,patterns):
        """ load space patterns of extral data rather than from solving

        Args:
            patterns (np.ndarray): shape = (npt, *nspace)
        """
        patterns_num = patterns.shape[0]
        self.patterns_num = patterns_num
        self.patterns = patterns
    
    def decoder(self,pcs,):
        """
        
        project pcs on patterns to get original fields

        """
        pcs_num = pcs.shape[0]
        time_num = pcs.shape[1]
        if pcs_num > self.patterns_num:
            raise ValueError(f"PC Number is {pcs_num}, larger than PT Number = {self.patterns_num}")
        # else:
        if self.pc_scaling == "std":
            # pcs = pcs * self.pc_std
            proj_pt = self.patterns[:pcs_num] * self.pc_std[:pcs_num]
        # projection use einsum
        # i: pcs_num, j: time
        # k...: *space patterns shape
        fields = np.einsum('ij,ik...->jk...', pcs, proj_pt).reshape((time_num, *self.origin_shape[1:]))
        # fields
        # fields: time, *space patterns shape
        return fields
    
    def decoder1(self,pcs,):
        """
        pcs: (*time, pc_num)
        project pcs on patterns to get original fields
        """
        pcs_num = pcs.shape[-1]
        time_shape = pcs.shape[:-1]
        pcs_shaped = pcs.reshape((-1,pcs_num)).swapaxes(0,1) # pc_num, time
        if pcs_num > self.patterns_num:
            raise ValueError(f"PC Number is {pcs_num}, larger than PT Number = {self.patterns_num}")
        # else:
        if self.pc_scaling == "std":
            # pcs = pcs * self.pc_std
            proj_pt = self.patterns[:pcs_num] * self.pc_std[:pcs_num]
        # projection use einsum
        # i: pcs_num, j: time
        # k...: *space patterns shape
        fields = np.einsum('ij,ik...->jk...', pcs_shaped, proj_pt).reshape((*time_shape, *self.origin_shape[1:]))
        if self.weights is not None:
            fields = fields / self.weights
        if self.channel_std is not None:
            fields = fields * self.channel_std # transform to original scale
        # fields
        # fields: time, *space patterns shape
        return fields



if __name__ == "__main__":
    import sacpy as scp
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    sst = scp.load_sst()["sst"].loc[:, -20:30, 150:275]
    ssta = scp.get_anom(sst)
    ssta_std = ssta.std()
    print(ssta_std)
    weights = np.sqrt(np.cos(np.deg2rad(ssta.lat))).to_numpy()[:, np.newaxis]

    eof = EOF(ssta/ssta_std,weights=weights,
              channel_std=ssta_std.to_numpy()
              )
    eof.solve(method="dask_svd",chunks=(1000),dim_min =30)
    # ev = eof.e_vector
    # print(eof.get_varperc())
    # print((ev @ ev.T).compute())
    # get pc and pattern
    pc = eof.get_pc()
    pt = eof.get_pt()
    eof.save("eof_test.pkl") 
    with open("eof_test.pkl", "rb") as f:
        eof = pickle.load(f)
    pc = eof.pc # npt,time
    pt = eof.pt
    # pc1 = pc.T.reshape(3,-1,pc.shape[0])
    new = eof.decoder1(pc.T) # time, *space
    print(new.shape)
    pseu_pc = eof.projection(new,npt=30)
    # print(pseu_pc.shape)
    print(np.max(np.abs(pseu_pc-pc)))
    # print()
    # print(np.corrcoef(pc1.reshape(3,-1),pseu_pc))

    # print(np.dot(pc,pc.T).compute())
    # pt = np.nan_to_num(eof.get_pt(npt=2,scaling=None).reshape(2,-1))
    # print(pt @ pt.T)
    # # get proprtion of mode variance
    # print(eof.get_varperc(npt=3))
    os._exit(0)
    import cartopy.crs as ccrs
    import sacpy.Map
    lon , lat = sst.lon , sst.lat
    # # =========================set figure================================
    fig = plt.figure(figsize=[9,7])
    # # =========================     ax  ================================
    ax = fig.add_subplot(221,projection=ccrs.PlateCarree(central_longitude=180))
    m1 = ax.scontourf(lon,lat,new[0,0],cmap='RdBu_r',
                      levels=np.linspace(-0.75,0.75,15)*2,
                    extend="both")
    ax.scontour(m1,colors="black")
    ax.init_map(smally=2.5)
    ax.set_title("re1")
    # # # =========================    ax2  ================================
    ax = fig.add_subplot(222,projection=ccrs.PlateCarree(central_longitude=180))
    m1 = ax.scontourf(lon,lat,new[0,10],cmap='RdBu_r',
                      levels=np.linspace(-0.75,0.75,15)*2,
                    extend="both")
    ax.scontour(m1,colors="black")
    ax.init_map(smally=2.5)
    ax.set_title("re2")
    # # # =========================    ax3  ================================
    ax3 = fig.add_subplot(223,projection=ccrs.PlateCarree(central_longitude=180))
    m2 = ax3.scontourf(lon,lat,ssta[0],cmap='RdBu_r',
                       levels=np.linspace(-0.75,0.75,15)*2,
                       extend="both")
    ax3.scontour(m2,colors="black")
    ax3.init_map(smally=2.5)
    ax3.set_title("o1")
    # # # =========================   ax4   ================================
    ax = fig.add_subplot(224,projection=ccrs.PlateCarree(central_longitude=180))
    m1 = ax.scontourf(lon,lat,ssta[10],cmap='RdBu_r',
                      levels=np.linspace(-0.75,0.75,15)*2,
                    extend="both")
    ax.scontour(m1,colors="black")
    ax.init_map(smally=2.5)
    ax.set_title("o2")
    # # # =========================  colorbar  ================================
    # cb_ax = fig.add_axes([0.1,0.03,0.4,0.02])
    # fig.colorbar(m1,cax=cb_ax,orientation="horizontal")
    plt.savefig("kk.png")


        


    # def pattern_corr(self, data: np.ndarray, npt=None):
    #     """  calculate pattern correlation of extra data and patterns

    #     Args:
    #         data (np.ndarray): shape (time, * space number)
    #         npt (int, optional): number of spatial patterns . Defaults to None.

    #     Returns:
    #         corr , p_value: corr and p_value shape (npt,data_time)
    #     """
    #     if npt is None:
    #         npt = self.dim_min
    #     # mask data
    #     data_noNan = data.reshape(data.shape[0], -1)[:, self.flag]
    #     # get need e_vector
    #     need_evctor = self.e_vector[:npt]
    #     # free degree
    #     N = need_evctor.shape[1] - 2
    #     # normalize data
    #     norm_evctor = (need_evctor - need_evctor.mean(axis=1)[..., np.newaxis]) / \
    #                   (need_evctor.std(axis=1)[..., np.newaxis])
    #     data_noNan_norm = (data_noNan - data_noNan.mean(axis=1)[..., np.newaxis]) / \
    #                       (data_noNan.std(axis=1)[..., np.newaxis])

    #     corr = norm_evctor @ data_noNan_norm.T / (N + 2)  # npatterns,data_time
    #     t_value = np.abs(corr / (EPS + np.sqrt(1 - corr**2)) * np.sqrt(N))
    #     p_value = sts.t.sf(t_value, df=N - 2) * 2
    #     return corr, p_value
