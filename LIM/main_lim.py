import os,sys
import slim.SNLIM
import yaml
import slim
import lim_utils as utils
import numpy as np
import pickle
from numpy.linalg import pinv, eigvals, eig, eigh


yml_path = sys.argv[1]

# open yml file
with open(yml_path, 'r') as stream: config = yaml.safe_load(stream)

main_path = config['main_path'] + config['name'] + "/"


if not os.path.exists(main_path):
    os.makedirs(main_path)
else:
    print("The folder already exists, please change the name in the yml file.")

os.system("cp {} {}".format(yml_path, main_path + "config.yml"))


logger = slim.get_logger(save_path=main_path + 'log.txt',mode='w')
var_infos = config['vars_info']

# build the model

pcs_ls,eofs = utils.load_pcs(config['build_dir'], var_infos)
logger.info("pcs_ls shape: {}".format(pcs_ls.shape))

if config.get("train_number") is not None:
    pcs_ls = pcs_ls[:config['train_number'],:]

if config['type'] == "LIM":
    LIM_model = slim.LIM(pcs_ls,fit_noise=True)
elif config['type'] == "CSLIM":
    cycle_ind = np.repeat(np.arange(1,5)[None,:],repeats=pcs_ls.shape[0] // 4 ,axis=0).flatten()
    LIM_model = slim.CSLIM(pcs_ls,cycle_ind=cycle_ind,fit_noise=True,cycle_labels=np.arange(1,5),logger=logger)
elif config['type'] == "SNLIM":
    cycle_ind = np.repeat(np.arange(1,5)[None,:],repeats=pcs_ls.shape[0] // 4,axis=0).flatten()
    cycle_ind = [cycle_ind, np.arange(1,5)[:pcs_ls.shape[0] % 4]]
    cycle_ind = np.concatenate(cycle_ind)
    print(cycle_ind)
    LIM_model = slim.SNLIM(pcs_ls,months=cycle_ind,nMonths=np.arange(1,5),)
else:
    raise ValueError("Model type not recognized")

# save the model

model_path = main_path + "model.pkl"

if config['type'] == "CSLIM":
    logger.info("Model time scale: {} years".format((- 1 / LIM_model.L_mul_eigs.real / 4)[:10]))
else:
    logger.info("Model time scale: {}".format(- 1 / LIM_model.L_eigs.real / 4)[:10])


LIM_model.save_precalib(model_path)
logger.info("Model saved at: {}".format(model_path))

# verify the model

verify_das = utils.load_verify(config['verify_dir'], var_infos, eofs, config['verify_number'],logger)

encoded_pcs = utils.encoder(verify_das, eofs, infos=var_infos, logger=logger)

# prediction

if config['type'] == "LIM":
    forecast_pcs = LIM_model.forecast(encoded_pcs,fcast_leads=np.arange(1,config['lead_time']+1))
elif config['type'] == "CSLIM":
    month0 = np.repeat(np.arange(1,5)[None,:],repeats=encoded_pcs.shape[0] // 4 ,axis=0).flatten()
    forecast_pcs = LIM_model.forecast(encoded_pcs,month0=month0,length=config['lead_time'],sep=True)
elif config['type'] == "SNLIM":
    month0 = np.repeat(np.arange(1,5)[None,:],repeats=encoded_pcs.shape[0] // 4 ,axis=0).flatten()
    month0 = [month0, np.arange(1,5)[:encoded_pcs.shape[0] % 4]]
    month0 = np.concatenate(month0)
    forecast_pcs = LIM_model.forecast(encoded_pcs,month0=month0,fcast_leads=config['lead_time'])

logger.info("Forecasted pcs shape: {}".format(forecast_pcs.shape))

decoded_forecast = utils.decoder(forecast_pcs, eofs, var_infos, logger)

# test the correlation

corrs = utils.test_correlation(verify_das, decoded_forecast, var_infos, logger)

pickle.dump(corrs, open(main_path + "corrs.pkl", "wb"))
logger.info("Correlations saved at: {}".format(main_path + "corrs.pkl"))

utils.plot_corrs(corrs, main_path, var_infos, logger)



