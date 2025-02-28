import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from lib.misc.EUOneDimVanillaCall import EUOneDimVanillaCall, EUExchangeCall
import munch, json, time, os
from lib.equations import portfolios as portfolios
from lib.Hure.systemHure import systemHure as systemHure
from lib.OSM.systemOSM import systemOSM as systemOSM


run_name = "CustomBlackScholesPortfolio_final_uniformEE_N100_R1"
# run_name = 'LargeBSPortfolio_paper_final_ChenWan_d20_sig0p25_bias_B1024_R24_N240'  # this is a very fine time grid, say N = 100
# rebalancing dates need to be a subset of these N_tes
model = ['systemOSM_theta0', 'systemOSM_theta0p5', 'systemOSM_theta1', 'systemHure_theta1']
model = ['systemOSM_theta1', 'systemHure_theta1']
# model = ['systemOSM_theta0', 'systemOSM_theta0p5']
model = ['systemOSM_theta0p5', 'systemHure_theta1']
model = ['systemOSM_theta0p5']

M = 1024
chunk_size = 1024
number_of_chunks = int(M / chunk_size)
if M / chunk_size != number_of_chunks:
    raise ValueError("M must be divisible by chunk_size")

output_dir = './logs/hedging/' + run_name + '/'
delta_hedge_dir = output_dir + '/delta_hedge'
for dir in [output_dir, delta_hedge_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)


run_folder = './logs/' + model[0] + '/' + run_name
with open(run_folder + '/config.json') as json_data_file:
    config = json.load(json_data_file)
config = munch.munchify(config)
config.eqn_config.d = config.eqn_config.d
tf.keras.backend.set_floatx(config.net_config.dtype)

BSDE = getattr(portfolios, config.eqn_config.eqn_name)(config.eqn_config)
if 'systemOSM' in model[0]:
    solver = systemOSM(config, BSDE)
elif 'systemHure' in model[0]:
    solver = systemHure(config, BSDE)
else:
    raise NotImplementedError("Model not recognized.")
solver.load_from_file(run_folder + "/trained_nets/")

hedging_setup = {"N": BSDE.N, "R": BSDE.R_list, "M": M,
                 "chunk_size": chunk_size, "T": BSDE.T, "d": BSDE.d, "J": BSDE.J, "m": BSDE.m}
hedging_setup = munch.munchify(hedging_setup)
with open(output_dir + '/setup.json', 'w') as f:
    json.dump(hedging_setup, f)

min_x = BSDE.x_0
max_x = BSDE.x_0
for chunk_number in range(number_of_chunks):
    dw_chunk, x_chunk = solver.bsde.sample(chunk_size)
    np.save(output_dir + 'dw_sample_chunk_' + str(chunk_number) + ".npy", dw_chunk)
    np.save(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy", x_chunk)

    x_scalar = np.prod(x_chunk ** (1 / BSDE.d), axis=1)
    min_x = np.min([np.min(x_scalar), min_x])
    max_x = np.max([np.max(x_scalar), max_x])

print("min=%.2f"%min_x)
print("max=%.2f"%max_x)
exit(0)
