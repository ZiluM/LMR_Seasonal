import os,sys
import yaml
import slim
import lim_utils as utils
import numpy as np
import pickle

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

cycle_ind = np.repeat(np.arange(1,5)[None,:],repeats=13000,axis=0).flatten()
CSLIM_model = slim.CSLIM(pcs_ls,cycle_ind=cycle_ind)

# save the model

model_path = main_path + "model.pkl"

CSLIM_model.save_precalib(model_path)
logger.info("Model saved at: {}".format(model_path))

# verify the model
verify_das = utils.load_verify(config['verify_dir'], var_infos, eofs, config['verify_number'],logger)

encoded_pcs = utils.encoder(verify_das, eofs, infos=var_infos, logger=logger)

# prediction

forecast_pcs = CSLIM_model.forecast(encoded_pcs,fcast_leads=np.arange(1,config['lead_time']+1),use_h5=False)