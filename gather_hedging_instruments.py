import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from scipy import sparse
from multiprocessing import Pool


from lib.misc.EUOneDimVanillaCall import EUOneDimVanillaCall, EUExchangeCall
import munch, json, time, os
from lib.equations import portfolios as portfolios
from lib.Hure.systemHure import systemHure as systemHure
from lib.OSM.systemOSM import systemOSM as systemOSM


run_name = 'LargeBSPortfolio__ItoTransposed_d50_sig0p25_N100_R1_K110'  # this is a very fine time grid, say N = 100
# rebalancing dates need to be a subset of these dates
model = ['systemOSM_theta0p5']

max_N_rebalance = 100
MODE = 'diagonal'
SEC = 'call-exchange'
T_TILDE = 4

output_dir = './logs/hedging/' + run_name + '/'
gamma_hedge_dir = output_dir + '/instruments/' + MODE + "_" + SEC + "_Ttilde" + str(T_TILDE) + '/'
if not os.path.exists(gamma_hedge_dir):
    os.makedirs(gamma_hedge_dir)

with open(output_dir + '/setup.json') as json_data_file:
    setup = json.load(json_data_file)
setup = munch.munchify(setup)
N_finest = setup.N
M = setup.M
chunk_size = setup.chunk_size
number_of_chunks = int(M / chunk_size)

if N_finest % max_N_rebalance != 0:
    raise ValueError

mod = int(N_finest / max_N_rebalance)



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

for chunk_number in range(number_of_chunks):
    dw_chunk, x_chunk = solver.bsde.sample(chunk_size)
    x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")

    weights_dir = gamma_hedge_dir + '/chunk_' + str(chunk_number)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    for n in range(max_N_rebalance + 1):
        n_subindx = mod * n


        security_n, \
            sparse_diff_security_n_v2, sparse_hess_security_n_v2\
            = solver.bsde.get_hedging_securities_at_t_n_sparse(n_subindx, x_chunk, MODE,
                                                               securities=SEC, K_ex=1, K_1d=solver.bsde.K,
                                                               T_tilde=T_TILDE)
        np.save(weights_dir + "/sec_sample_n_" + str(n_subindx) + ".npy", security_n)
        sparse.save_npz(weights_dir + "/sparse_dk_sec_sample_n_" + str(n_subindx) + ".npz", sparse.csr_array(sparse_diff_security_n_v2))
        sparse.save_npz(weights_dir + "/sparse_dlk_sec_sample_n_" + str(n_subindx) + ".npz", sparse.csr_array(sparse_hess_security_n_v2))


exit(0)
