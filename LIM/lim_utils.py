import slim
import numpy as np
import os,sys
import pickle
sys.path.append("/glade/work/zilumeng/SSNLIM/")
import EOF
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sacpy.Map



num_workers = 8

def load_pcs(build_dir,vars_info):
    pc_ls = []
    eof_ls = []
    for var_name,info in vars_info.items():
        eof_path = build_dir + info['file_name']
        pc_num = info['num']
        eof = pickle.load(open(eof_path, 'rb'))
        pc = eof.pc[:pc_num]
        pc_ls.append(pc)
        eof_ls.append(eof)
    pc_ls = np.concatenate(pc_ls,axis=0) # pc_num, train_num
    pc_ls = pc_ls.T # train_num, pc_num
    return pc_ls.real,eof_ls

def load_xarray(path,name,slice0,domain='global'):
    da = xr.open_dataset(path)[name][slice0]
    if domain == 'NH':
        da = da.sel(lat=slice(0,90))
    return da

def load_verify(verif_dir,vars_info,eofs,verif_number,logger=None):
    if logger is not None:
        logger.info("Loading verification data")
    else:
        logger = slim.get_logger()
    if len(verif_number) == 1:
        verif_slice = slice(int(verif_number[0]),None)
    else:
        verif_slice = slice(int(verif_number[0]),int(verif_number[1]))
    das = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for var_name,info in vars_info.items():
            verif_path = verif_dir + info['verify_name']
            da = executor.submit(load_xarray,verif_path,var_name,verif_slice,info['domain'])
            das.append(da)
            # logger.info("Loaded {}".format(var_name))
        das = [da.result() for da in das]
        shapes = [da.shape for da in das]
        logger.info("Verification data shapes: {}".format(shapes))
    return das

def encoder(das,eofs,infos,logger=None):
    if logger is not None:
        logger.info("Encoding verification data")
    else:
        logger = slim.get_logger()
        logger.info("Encoding verification data")
    encoded = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for da,eof,info in zip(das,eofs,infos.values()):
            if bool(info.get("season_norm")):
                season_idx = np.repeat(np.arange(1,5)[None,:],repeats=da.shape[0]//4,axis=0).flatten()
                features = executor.submit(eof.projection1,da,npt=info['num'],time_dims=1,season_idx=season_idx)
            else:
                features = executor.submit(eof.projection1,da,npt=info['num'],time_dims=1)
            encoded.append(features)
        encoded = [feature.result() for feature in encoded]
        shapes = [feature.shape for feature in encoded]
        logger.info("Encoded data shapes: {}".format(shapes))
        encoded = np.concatenate(encoded,axis=1) # test_num, feature_num
        logger.info("Encoded data shape after concatenation: {}".format(encoded.shape))
    return encoded


def sperate_pcs(pcs,infos):
    pcs_dict = {}
    for var_name,info in infos.items():
        pc = pcs[...,:info['num']]
        pcs = pcs[...,info['num']:]
        pcs_dict[var_name] = pc
    return pcs_dict



def decoder(pcs,eofs,infos,logger=None):
    if logger is not None:
        logger.info("Decoding forecast data")
    else:
        logger = slim.get_logger()
        logger.info("Decoding forecast data")
    decoded = []
    # sperate pcs into different variables
    pcs_ls = []
    for var_name,info in infos.items():
        pc = pcs[...,:info['num']]
        pcs = pcs[...,info['num']:]
        pcs_ls.append(pc)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for pc,eof,info in zip(pcs_ls,eofs,infos.values()):
            if bool(info.get("season_norm")):
                season_idx = np.repeat(np.arange(1,5)[None,:],repeats=pc.shape[0]//4,axis=0).flatten()
                da = executor.submit(eof.decoder1,pc,season_idx=season_idx)
            else:
                da = executor.submit(eof.decoder1,pc)
            decoded.append(da)
        decoded = [da.result() for da in decoded]
        shapes = [da.shape for da in decoded]
        logger.info("Decoded data shapes: {}".format(shapes))
    return decoded

def decoder_seq(pcs,eofs,infos,months=None,logger=None):
    """
    pcs: time, feature_num
    eofs: list of eofs
    infos: dict of vars_info
    months: list of months of pcs [1,4,7,10,...]
    logger: logger
    """
    if logger is not None:
        logger.info("Decoding forecast data")
    else:
        logger = slim.get_logger()
        logger.info("Decoding forecast data")
    decoded = []
    # sperate pcs into different variables
    pcs_ls = []
    for var_name,info in infos.items():
        pc = pcs[...,:info['num']]
        pcs = pcs[...,info['num']:]
        pcs_ls.append(pc)
    for pc,eof,info in zip(pcs_ls,eofs,infos.values()):
        if bool(info.get("season_norm")):
            if months is None:
                season_idx = np.repeat(np.arange(1,5)[None,:],repeats=pc.shape[0]//4,axis=0).flatten()
            else:
                # season_idx = months
                months_new = np.copy(months)
                months_new[months == 1] = 1
                months_new[months == 4] = 2
                months_new[months == 7] = 3
                months_new[months == 10] = 4
                season_idx = months_new
                logger.info("Using specific months for season normalization")
            da = eof.decoder1(pc,season_idx=season_idx)
        else:
            da = eof.decoder1(pc)
        decoded.append(da)
    # decoded = [da.result() for da in decoded]
    shapes = [da.shape for da in decoded]
    logger.info("Decoded data shapes: {}".format(shapes))
    return decoded

def decoder_specific_locations(pcs,eofs,infos,var_obs,locations):
    # sperate pcs into different variables
    pcs_dc = {}
    eofs_dc = {}
    for var_name,info in infos.items():
        pc = pcs[...,:info['num']]
        pcs = pcs[...,info['num']:]
        pcs_dc[var_name] = pc
    for var_name,eof in zip(list(infos.keys()),eofs):
        eofs_dc[var_name] = eof
    
    used_pc = pcs_dc[var_obs]
    used_eof = eofs_dc[var_obs]
    decoded_point = used_eof.decoder_point(used_pc,locations) # time num
    return decoded_point


    


def test_correlation(verify_das,decoded_forecast,infos,logger=None):
    if logger is not None:
        logger.info("Testing correlation")
    else:
        logger = slim.get_logger()
        logger.info("Testing correlation")
    corrs = []
    # for lead in range(decoded_forecast.shape[0]):
    corr_dict = {}
    for real_da, forecast_da, name in zip(verify_das,decoded_forecast,infos.keys()):
        corr_ls = []
        for lead in range(1,forecast_da.shape[0]+1):
            corr = slim.field_corr(real_da[lead:],forecast_da[lead-1,:-lead])
            corr_ls.append(corr)
        corr_dict[name] = np.array(corr_ls)
    # logger.info("Correlation results: {}".format(corr_dict))
    for name,corr in corr_dict.items():
        logger.info("Correlation results for {}: {}".format(name,corr.shape))
    return corr_dict


def init_map(info,fig=None):
    if fig is None:
        fig = plt.figure()
    if info['domain'] == 'global':
        proj = ccrs.Robinson(central_longitude=180)
    elif info['domain'] == 'NH':
        proj = ccrs.NorthPolarStereo(central_longitude=180)

    ax = fig.add_subplot(111,projection=proj)
    ax.coastlines()
    if info['domain'] == 'global':
        ax.set_global()
    elif info['domain'] == 'NH':
        ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
    lon = np.arange(0,360,2)
    lat = np.arange(-89,90,2)
    if info['domain'] == 'NH':
        # lon = np.arange(0,360,2)
        lat = lat[lat>=0]
    return fig,ax,lon,lat
        

def plot_corrs(corrs,main_path,infos,logger=None):
    if logger is not None:
        logger.info("Plotting correlation")
    else:
        logger = slim.get_logger()
        logger.info("Plotting correlation")
    if not os.path.exists(main_path + "pic/"):
        os.makedirs(main_path + "pic/")
    else:
        logger.info("The folder already exists, please change the name in the yml file.")
    for name,corr in corrs.items():
        for lead in range(corr.shape[0]): 
            fig,ax,lon,lat = init_map(infos[name])
            area_weight = np.sqrt(np.cos(np.deg2rad(lat)))
            area_weight = np.repeat(area_weight[:,np.newaxis],lon.shape[0],axis=1)
            used_weight = area_weight[np.isnan(corr[lead])==False]
            mean_corr = np.nanmean(corr[lead]*area_weight)/np.nanmean(used_weight)
            m = ax.scontourf(lon,lat,corr[lead],transform=ccrs.PlateCarree(),levels=np.arange(-1,1.1,0.1),cmap='RdBu_r')
            ax.init_map()
            ax.set_title("Lead time: {}, var: {}, mean: {:.2f}".format(lead+1,name,mean_corr))
            fig.colorbar(m,ax=ax,orientation='horizontal',shrink=0.8)
            fig.savefig(main_path + "pic/{}_lead{}.png".format(name,lead+1))
            logger.info("Saved at: {}".format(main_path + "pic/{}_lead{}.png".format(name,lead+1)))

        # for lead in range(corr.
        
        
    
    
    

    