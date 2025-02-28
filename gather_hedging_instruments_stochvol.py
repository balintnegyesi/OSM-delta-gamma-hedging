import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np


def _get_closest_idx(array, value):
    # # # finds the index of the element in array which is closest to value
    arr = np.asarray(array)
    idx = (np.abs(arr - value)).argmin()
    return idx

from lib.misc.EUOneDimVanillaCall import EUOneDimVanillaCall, EUExchangeCall
import munch, json, time, os
from lib.equations import portfolios as portfolios
from lib.equations import backward_equations_arch as eqns
from lib.Hure.systemHure import systemHure as systemHure
from lib.OSM.systemOSM import systemOSM as systemOSM


run_name = 'BermudanGeometricHeston_R10_N50_euler_tranposed_ito_long'  # this is a very fine time grid, say N = 100
max_N_rebalance = 50
# rebalancing dates need to be a subset of these dates
model = ['systemOSM_theta0p5']
run_folder = './logs/' + model[0] + '/' + run_name
with open(run_folder + '/config.json') as json_data_file:
    config = json.load(json_data_file)
config = munch.munchify(config)
config.eqn_config.d = config.eqn_config.d
tf.keras.backend.set_floatx(config.net_config.dtype)
BSDE = getattr(portfolios, config.eqn_config.eqn_name)(config.eqn_config)


# # ------------------------------------------------------------------------------------------------------------------
vega_model = 'systemOSM_theta0p5'
vega_reference_run = 'BermudanGeometricHeston_SetA_tranposed_instrument_vega'
with open('./logs/' + vega_model + '/' + vega_reference_run + '/config.json') as json_data_file:
    config_vega = json.load(json_data_file)
config_vega = munch.munchify(config_vega)
tf.keras.backend.set_floatx(config_vega.net_config.dtype)
vega_refBSDE = getattr(portfolios, config_vega.eqn_config.eqn_name)(config_vega.eqn_config)
if "OSM" in vega_model:
    vega_solver = systemOSM(config_vega, vega_refBSDE)
elif "Hure" in vega_model:
    vega_solver = systemHure(config_vega, vega_refBSDE)
else:
    raise NotImplementedError()
vega_solver.load_from_file('./logs/' + vega_model + '/' + vega_reference_run + "/trained_nets/")
# # ------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------
gamma_model = 'systemOSM_theta0p5'
gamma_reference_run = 'BermudanGeometricHeston_SetA_tranposed_instrument_gamma'
with open('./logs/' + gamma_model + '/' + gamma_reference_run + '/config.json') as json_data_file:
    config_gamma = json.load(json_data_file)
config_gamma = munch.munchify(config_gamma)
tf.keras.backend.set_floatx(config_gamma.net_config.dtype)
gamma_refBSDE = getattr(portfolios, config_gamma.eqn_config.eqn_name)(config_gamma.eqn_config)
gamma_solver = systemOSM(config_gamma, gamma_refBSDE)
gamma_solver.load_from_file('./logs/' + gamma_model + '/' + gamma_reference_run + "/trained_nets/")
# # ------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------
vomma_model = 'systemOSM_theta0p5'
vomma_reference_run = 'BermudanGeometricHeston_SetA_tranposed_instrument_vomma'
with open('./logs/' + vomma_model + '/' + vomma_reference_run + '/config.json') as json_data_file:
    config_vomma = json.load(json_data_file)
config_vomma = munch.munchify(config_vomma)
tf.keras.backend.set_floatx(config_vomma.net_config.dtype)
vomma_refBSDE = getattr(portfolios, config_vomma.eqn_config.eqn_name)(config_vomma.eqn_config)
vomma_solver = systemOSM(config_vomma, vomma_refBSDE)
vomma_solver.load_from_file('./logs/' + vomma_model + '/' + vomma_reference_run + "/trained_nets/")
# # ------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------
vanna_model = 'systemOSM_theta0p5'
vanna_reference_run = 'BermudanGeometricHeston_SetA_tranposed_instrument_vanna'
with open('./logs/' + vanna_model + '/' + vanna_reference_run + '/config.json') as json_data_file:
    config_vanna = json.load(json_data_file)
config_vanna = munch.munchify(config_vanna)
tf.keras.backend.set_floatx(config_vanna.net_config.dtype)
vanna_refBSDE = getattr(portfolios, config_vanna.eqn_config.eqn_name)(config_vanna.eqn_config)
vanna_solver = systemOSM(config_vanna, vanna_refBSDE)
vanna_solver.load_from_file('./logs/' + vanna_model + '/' + vanna_reference_run + "/trained_nets/")
# # ------------------------------------------------------------------------------------------------------------------


output_dir = './logs/hedging/' + run_name + '/'
instrument_dir = output_dir + '/instruments/'
if not os.path.exists(instrument_dir):
    os.makedirs(instrument_dir)

with open(output_dir + '/setup.json') as json_data_file:
    setup = json.load(json_data_file)
setup = munch.munchify(setup)
N_finest = setup.N
M = setup.M
chunk_size = setup.chunk_size
number_of_chunks = int(M / chunk_size)
max_N_rebalance = N_finest

if N_finest % max_N_rebalance != 0:
    raise ValueError

if N_finest / max_N_rebalance != int(N_finest / max_N_rebalance):
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
    # dw_chunk, x_chunk = solver.bsde.sample(chunk_size)
    x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")
    print(x_chunk.shape)

    weights_dir = instrument_dir + '/chunk_' + str(chunk_number)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    num_vega = solver.bsde.d - solver.bsde.m  # number of stoch vols
    num_gamma = solver.bsde.m  # number of assets
    num_vomma = 1 * num_vega
    num_vanna = 1

    if num_vega != 1:
        raise NotImplementedError("Only one-dimensional Heston is implemented so far")
    if num_gamma != 1:
        raise NotImplementedError("Only one-dimensional Heston is implemented so far")
    K = num_vega + num_gamma + num_vanna + num_vomma
    # K = 1
    sec_sample = np.zeros([chunk_size, K, solver.bsde.N + 1]).astype(tf.keras.backend.floatx())
    diff_sec_sample = np.zeros([chunk_size, solver.bsde.d, K, solver.bsde.N + 1]).astype(tf.keras.backend.floatx())
    hess_sec_sample = np.zeros([chunk_size, solver.bsde.d, solver.bsde.d, K, solver.bsde.N + 1]).astype(tf.keras.backend.floatx())

    for n in range(max_N_rebalance + 1):
        n_subindx = mod * n
        # # #
        t_n = solver.bsde.t[mod * n]
        x_n = x_chunk[..., mod * n]
        # get vega instrument first
        # # ------------------------------------------------------------------------------------------------------------------
        n_vega = _get_closest_idx(vega_refBSDE.t, t_n)
        t_n_vega = vega_refBSDE.t[n_vega]
        print("t_n=%.4f, t_n_vega=%.4f, diff=%.2e"%(t_n, t_n_vega, t_n-t_n_vega))
        c_n = vega_solver.Y[n_vega](x_n, False)
        z_tilde_n = vega_solver.Z[n_vega](x_n, False)
        try:
            if n_vega < vega_solver.bsde.N:
                if "OSM" in vega_model:
                    gamma_n = vega_solver.G[n_vega](x_n, False)
                elif "Hure" in vega_model:
                    x_tf = tf.constant((x_n))
                    with tf.GradientTape() as tape:
                        tape.watch(x_tf)
                        z_n = vega_solver.Z[n_vega](x_tf, False)
                    gamma_n = tape.batch_jacobian(z_n, x_tf)
                else:
                    raise NotImplementedError
                gamma_n = tf.transpose(gamma_n, perm=[0, 2, 3, 1])
            # if "OSM" not in vega_model:
            #     raise NotImplementedError
        except:
            raise ValueError
            # gamma_n = np.zeros(shape=[chunk_size, solver.bsde.d, solver.bsde.d, K])
        l_n = vega_solver.bsde.g_tf(vega_solver.bsde.T, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ",
                                 tf.where(l_n > c_n, 1.0, 0.0), vega_solver.bsde.is_exercise_date[..., n_vega])
        y_n = c_n + is_exercised * (l_n - c_n)

        grad_l_n = vega_solver.bsde.gradx_g_tf(vega_solver.bsde.T, x_n)
        sigma_n = vega_solver.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", grad_l_n, sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        inv_sigma_n = vega_solver.bsde.inverse_sigma_process_tf(t_n, x_n)
        nabla_y_n = tf.einsum("MJd, Mdq -> MqJ", z_n, inv_sigma_n)

        if n_vega < vega_solver.bsde.N:
            nabla_u_nabla_sigma = tf.einsum("MdJ, Mdqm -> MqmJ",
                                            nabla_y_n, vega_solver.bsde.nabla_sigma_process_tf(t_n, x_n))

            hess_y_n = tf.einsum("Mij, MikJ -> MjkJ", inv_sigma_n, gamma_n - nabla_u_nabla_sigma)
        else:
            hess_y_n = tf.transpose(vega_solver.bsde.hessx_g_tf(t_n, x_n), perm=[0, 2, 3, 1])
        t_grads = time.time()
        sec_sample[..., 0:1, n] = y_n
        diff_sec_sample[..., 0:1, n] = nabla_y_n
        hess_sec_sample[..., 0:1, n] = hess_y_n
        # # ------------------------------------------------------------------------------------------------------------------

        # get gamma instrument
        # # ------------------------------------------------------------------------------------------------------------------
        n_gamma = _get_closest_idx(gamma_refBSDE.t, t_n)
        t_n_gamma = gamma_refBSDE.t[n_gamma]
        print("t_n=%.4f, t_n_gamma=%.4f, diff=%.2e"%(t_n, t_n_gamma, t_n-t_n_gamma))
        c_n = gamma_solver.Y[n_gamma](x_n, False)
        z_tilde_n = gamma_solver.Z[n_gamma](x_n, False)
        try:
            if n_gamma < gamma_solver.bsde.N:
                if "OSM" in gamma_model:
                    gamma_n = gamma_solver.G[n_gamma](x_n, False)
                elif "Hure" in gamma_model:
                    x_tf = tf.constant((x_n))
                    with tf.GradientTape() as tape:
                        tape.watch(x_tf)
                        z_n = gamma_solver.Z[n_gamma](x_tf, False)
                    gamma_n = tape.batch_jacobian(z_n, x_tf)
                else:
                    raise NotImplementedError
                gamma_n = tf.transpose(gamma_n, perm=[0, 2, 3, 1])
        except:
            raise ValueError
        l_n = gamma_solver.bsde.g_tf(gamma_solver.bsde.T, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ",
                                 tf.where(l_n > c_n, 1.0, 0.0), gamma_solver.bsde.is_exercise_date[..., n_gamma])
        y_n = c_n + is_exercised * (l_n - c_n)

        grad_l_n = gamma_solver.bsde.gradx_g_tf(gamma_solver.bsde.T, x_n)
        sigma_n = gamma_solver.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", grad_l_n, sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        inv_sigma_n = gamma_solver.bsde.inverse_sigma_process_tf(t_n, x_n)
        # inv_sigmaT_n = gamma_solver.bsde.inverse_sigma_transpose_process_tf(t_n, x_n)
        # sum_of_diff = np.sum(tf.transpose(inv_sigma_n, perm=[0, 2, 1]) != inv_sigmaT_n)
        # if sum_of_diff > 0:
        #     raise ValueError("inverse sigma^T is not the same as (inverse sigma)^T num_diff=%d"%sum_of_diff)
        nabla_y_n = tf.einsum("MJd, Mdq -> MqJ", z_n, inv_sigma_n)

        if n_gamma < gamma_solver.bsde.N:
            nabla_u_nabla_sigma = tf.einsum("MdJ, Mdqm -> MqmJ",
                                            nabla_y_n, gamma_solver.bsde.nabla_sigma_process_tf(t_n, x_n))

            hess_y_n = tf.einsum("Mij, MikJ -> MjkJ", inv_sigma_n, gamma_n - nabla_u_nabla_sigma)
        else:
            hess_y_n = tf.transpose(gamma_solver.bsde.hessx_g_tf(t_n, x_n), perm=[0, 2, 3, 1])

        t_grads = time.time()
        sec_sample[..., 1:2, n] = y_n
        diff_sec_sample[..., 1:2, n] = nabla_y_n
        hess_sec_sample[..., 1:2, n] = hess_y_n
        # # ------------------------------------------------------------------------------------------------------------------

        # get vomma instrument
        # # ------------------------------------------------------------------------------------------------------------------
        n_vomma = _get_closest_idx(vomma_refBSDE.t, t_n)
        t_n_vomma = vomma_refBSDE.t[n_vomma]
        print("t_n=%.4f, t_n_gamma=%.4f, diff=%.2e" % (t_n, t_n_vomma, t_n - t_n_vomma))
        c_n = vomma_solver.Y[n_vomma](x_n, False)
        z_tilde_n = vomma_solver.Z[n_vomma](x_n, False)
        try:
            if n_vomma < vomma_solver.bsde.N:
                if "OSM" in vomma_model:
                    gamma_n = vomma_solver.G[n_vomma](x_n, False)
                elif "Hure" in vomma_model:
                    x_tf = tf.constant((x_n))
                    with tf.GradientTape() as tape:
                        tape.watch(x_tf)
                        z_n = vomma_solver.Z[n_vomma](x_tf, False)
                    gamma_n = tape.batch_jacobian(z_n, x_tf)
                else:
                    raise NotImplementedError
                gamma_n = tf.transpose(gamma_n, perm=[0, 2, 3, 1])
            # if "OSM" not in gamma_model:
            #     raise NotImplementedError
        except:
            raise ValueError
            # gamma_n = np.zeros(shape=[chunk_size, solver.bsde.d, solver.bsde.d, K])
        l_n = vomma_solver.bsde.g_tf(vomma_solver.bsde.T, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ",
                                 tf.where(l_n > c_n, 1.0, 0.0), vomma_solver.bsde.is_exercise_date[..., n_vomma])
        y_n = c_n + is_exercised * (l_n - c_n)

        grad_l_n = vomma_solver.bsde.gradx_g_tf(vomma_solver.bsde.T, x_n)
        sigma_n = vomma_solver.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", grad_l_n, sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        inv_sigma_n = vomma_solver.bsde.inverse_sigma_process_tf(t_n, x_n)
        nabla_y_n = tf.einsum("MJd, Mdq -> MqJ", z_n, inv_sigma_n)

        if n_vomma < vomma_solver.bsde.N:
            nabla_u_nabla_sigma = tf.einsum("MdJ, Mdqm -> MqmJ",
                                            nabla_y_n, vomma_solver.bsde.nabla_sigma_process_tf(t_n, x_n))

            hess_y_n = tf.einsum("Mij, MikJ -> MjkJ", inv_sigma_n, gamma_n - nabla_u_nabla_sigma)
        else:
            hess_y_n = tf.transpose(vomma_solver.bsde.hessx_g_tf(t_n, x_n), perm=[0, 2, 3, 1])

        t_grads = time.time()
        sec_sample[..., 2:3, n] = y_n
        diff_sec_sample[..., 2:3, n] = nabla_y_n
        hess_sec_sample[..., 2:3, n] = hess_y_n
        # # ------------------------------------------------------------------------------------------------------------------

        # get vanna instrument
        # # ------------------------------------------------------------------------------------------------------------------
        n_vanna = _get_closest_idx(vanna_refBSDE.t, t_n)
        t_n_vanna = vanna_refBSDE.t[n_vanna]
        print("t_n=%.4f, t_n_gamma=%.4f, diff=%.2e" % (t_n, t_n_vanna, t_n - t_n_vanna))
        c_n = vanna_solver.Y[n_vanna](x_n, False)
        z_tilde_n = vanna_solver.Z[n_vanna](x_n, False)
        try:
            if n_vanna < vanna_solver.bsde.N:
                if "OSM" in vomma_model:
                    gamma_n = vanna_solver.G[n_vanna](x_n, False)
                elif "Hure" in vomma_model:
                    x_tf = tf.constant((x_n))
                    with tf.GradientTape() as tape:
                        tape.watch(x_tf)
                        z_n = vanna_solver.Z[n_vanna](x_tf, False)
                    gamma_n = tape.batch_jacobian(z_n, x_tf)
                else:
                    raise NotImplementedError
                gamma_n = tf.transpose(gamma_n, perm=[0, 2, 3, 1])
            # if "OSM" not in gamma_model:
            #     raise NotImplementedError
        except:
            raise ValueError
            # gamma_n = np.zeros(shape=[chunk_size, solver.bsde.d, solver.bsde.d, K])
        l_n = vanna_solver.bsde.g_tf(vanna_solver.bsde.T, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ",
                                 tf.where(l_n > c_n, 1.0, 0.0), vanna_solver.bsde.is_exercise_date[..., n_vanna])
        y_n = c_n + is_exercised * (l_n - c_n)

        grad_l_n = vanna_solver.bsde.gradx_g_tf(vanna_solver.bsde.T, x_n)
        sigma_n = vanna_solver.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", grad_l_n, sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        inv_sigma_n = vanna_solver.bsde.inverse_sigma_process_tf(t_n, x_n)
        nabla_y_n = tf.einsum("MJd, Mdq -> MqJ", z_n, inv_sigma_n)

        if n_vomma < vanna_solver.bsde.N:
            nabla_u_nabla_sigma = tf.einsum("MdJ, Mdqm -> MqmJ",
                                            nabla_y_n, vanna_solver.bsde.nabla_sigma_process_tf(t_n, x_n))

            hess_y_n = tf.einsum("Mij, MikJ -> MjkJ", inv_sigma_n, gamma_n - nabla_u_nabla_sigma)
        else:
            hess_y_n = tf.transpose(vanna_solver.bsde.hessx_g_tf(t_n, x_n), perm=[0, 2, 3, 1])

        t_grads = time.time()
        sec_sample[..., 3:4, n] = y_n
        diff_sec_sample[..., 3:4, n] = nabla_y_n
        hess_sec_sample[..., 3:4, n] = hess_y_n
        # # ------------------------------------------------------------------------------------------------------------------


    np.save(weights_dir + "/sec_sample.npy", sec_sample)
    np.save(weights_dir + "/dk_sec_sample.npy", diff_sec_sample)
    np.save(weights_dir + "/dlk_sec_sample.npy", hess_sec_sample)




exit(0)
