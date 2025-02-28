import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from lib.equations import portfolios as portfolios
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew as scpskew
import dataframe_image as dfi


# from gamma_hedging_pnl_black_scholes.py import get_pnl_statistics
def get_pnl_statistics(pnl_sample):
    # # # shape: M x 1 x N + 1
    # # # returns: 1 x N + 1 shaped tensors corresponding to appropriate statistics
    retval = {'mean': np.mean(pnl_sample, axis=0), 'median': np.median(pnl_sample, axis=0),
              'variance': np.var(pnl_sample, axis=0), \
              'var01': np.quantile(pnl_sample, q=0.01, axis=0), 'var05': np.quantile(pnl_sample, q=0.05, axis=0), \
              'var95': np.quantile(pnl_sample, q=0.95, axis=0), 'var99': np.quantile(pnl_sample, q=0.99, axis=0), \
              'skew': scpskew(pnl_sample, axis=0)}

    retval['es01'] = np.zeros(retval['var01'].shape)
    retval['es05'] = np.zeros(retval['var05'].shape)
    retval['es95'] = np.zeros(retval['var95'].shape)
    retval['es99'] = np.zeros(retval['var99'].shape)
    retval['semivariance'] = np.zeros(retval['var01'].shape)

    for n in range(pnl_sample.shape[-1]):
        pnl_n = pnl_sample[..., n]
        retval['es01'][..., n] = np.mean(pnl_n[pnl_n <= retval['var01'][..., n]])
        retval['es05'][..., n] = np.mean(pnl_n[pnl_n <= retval['var05'][..., n]])
        retval['es95'][..., n] = np.mean(pnl_n[pnl_n >= retval['var95'][..., n]])
        retval['es99'][..., n] = np.mean(pnl_n[pnl_n >= retval['var99'][..., n]])
        retval['semivariance'][..., n] = np.var(pnl_n[pnl_n < retval['mean'][..., n]])

    return retval



from lib.Hure.systemHure import systemHure as systemHure
from lib.OSM.systemOSM import systemOSM as systemOSM
import munch, json, time, os


run_name = 'CustomBlackScholesPortfolio__final_varyingEE_N100'  # this is a very fine time grid, say N = 100
# rebalancing dates need to be a subset of these dates
model = ['systemOSM_theta0p5']

N_rebalance_list = [1, 2, 5, 10, 20, 100]

pnl_metrics = pd.DataFrame(columns=model,
                           index=['mean', 'median', 'variance', 'var01', 'var05', 'var95', 'var99', 'es01', 'es05',
                                  'es95', 'es99', 'semivariance', 'skew'])


output_dir = './logs/hedging/' + run_name + '/'
delta_hedge_dir = output_dir + '/delta_hedge/'

with open(output_dir + '/setup.json') as json_data_file:
    setup = json.load(json_data_file)
setup = munch.munchify(setup)
N_finest = setup.N
M = setup.M
chunk_size = setup.chunk_size
number_of_chunks = int(M / chunk_size)

for m in model:
    if 'OSM' in m:
        is_z_tilde = True
    elif "Hure" in m:
        is_z_tilde = False
    else:
        raise ValueError

    mean_list = []
    median_list = []
    variance_list = []
    run_folder = './logs/' + m + '/' + run_name
    with open(run_folder + '/config.json') as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    config.eqn_config.d = config.eqn_config.d
    tf.keras.backend.set_floatx(config.net_config.dtype)

    BSDE = getattr(portfolios, config.eqn_config.eqn_name)(config.eqn_config)
    if 'systemOSM' in m:
        solver = systemOSM(config, BSDE)
    elif 'systemHure' in m:
        solver = systemHure(config, BSDE)
    else:
        raise NotImplementedError("Model not recognized.")
    solver.load_from_file(run_folder + "/trained_nets/")


    for N_rebalance in N_rebalance_list:
        if N_finest % N_rebalance != 0:
            raise ValueError
        mod = int(N_finest / N_rebalance)

        results_dir = delta_hedge_dir + m + "/N_rebalance=" + str(N_rebalance)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # # # we first need to get the weights
        for chunk_number in range(number_of_chunks):
            print("Starting chunk #%d/%d" % (chunk_number + 1, number_of_chunks))
            dw_chunk = np.load(output_dir + 'dw_sample_chunk_' + str(chunk_number) + ".npy")
            x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")
            exercise_index_chunk = solver.find_exercise_index(x_chunk, N_rebalance)

            weights_dir = results_dir + '/chunk_' + str(chunk_number) + "/"
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            np.save(weights_dir + "/exercise_index.npy", exercise_index_chunk)

            for n in range(N_rebalance + 1):
                n_subidx = int(n * mod)
                print("n=%d/%d" % (n, N_rebalance))
                alpha_chunk_n, y_chunk_n = solver.delta_hedging_black_scholes_alpha_n(n_subidx, x_chunk[..., n_subidx], is_z_tilde,
                                                                                      exercise_index_chunk)
                np.save(weights_dir + "/alpha_h_n_" + str(n) + ".npy", alpha_chunk_n)
                np.save(weights_dir + "/y_n_" + str(n) + ".npy", y_chunk_n)
        # # # hedging weights gathered




        pnl_delta_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        relative_pnl_delta_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        exercise_index = np.zeros(shape=[M, solver.bsde.J], dtype=tf.keras.backend.floatx())

        for chunk_number in range(number_of_chunks):
            print("Starting chunk #%d/%d" % (chunk_number + 1, number_of_chunks))
            dw_chunk = np.load(output_dir + 'dw_sample_chunk_' + str(chunk_number) + ".npy")
            x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")

            weights_dir = results_dir + '/chunk_' + str(chunk_number) + "/"
            exercise_index_chunk = np.load(weights_dir + "/exercise_index.npy")

            pnl_delta_chunk, \
                relative_pnl_delta_chunk = solver.delta_hedging_black_scholes_portfolio(x_chunk, weights_dir,
                                                                                        N_rebalance,
                                                                                        exercise_index_chunk)


            idx_low = chunk_number * chunk_size
            idx_high = (chunk_number + 1) * chunk_size

            pnl_delta_sample[idx_low: idx_high, ...] = pnl_delta_chunk
            relative_pnl_delta_sample[idx_low: idx_high, ...] = relative_pnl_delta_chunk
            exercise_index[idx_low: idx_high, :] = exercise_index_chunk

        results_dir = delta_hedge_dir + m + "/N_rebalance=" + str(N_rebalance)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.save(results_dir + "/pnl_delta_sample.npy", pnl_delta_sample)
        np.save(results_dir + "/relative_pnl_delta_sample.npy", relative_pnl_delta_sample)
        np.save(results_dir + "/exercise_index.npy", exercise_index)

        mod = N_finest / N_rebalance
        exercise_date_rebalance = np.ceil(exercise_index / mod)

        rel_pnl_T = np.zeros((M, 1))
        for path_num in range(M):
            rel_pnl_T[path_num, 0] = relative_pnl_delta_sample[path_num, 0, int(exercise_date_rebalance[path_num, 0])]

        rel_pnl_T = relative_pnl_delta_sample[..., -1]

        print("N_rebalance=%d"%N_rebalance)
        mean = np.mean(rel_pnl_T)
        median = np.median(rel_pnl_T)
        variance = np.var(rel_pnl_T)
        print("mean=%.2e"%mean)
        print("median=%.2e"%median)
        print("variance=%.2e"%variance)
        print("\n\n")

        mean_list.append(mean)
        median_list.append(median)
        variance_list.append(variance)

        fig, ax = plt.subplots(dpi=200)
        ax.hist(rel_pnl_T, density=True, bins=20)
        ax.set_xlabel("relative PnL")
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 8])
        plt.savefig(results_dir + "/PnL.png")
        plt.close()

        pnl_metrics[m] = get_pnl_statistics(rel_pnl_T)
        dfi.export(pnl_metrics, results_dir + '/pnl_metrics.png')

    fig, ax = plt.subplots(dpi=200)
    empord, b = np.polyfit(np.log(N_rebalance_list), np.log(np.abs(mean_list)), 1)
    ax.scatter(N_rebalance_list, np.abs(mean_list), label='mean, O(N^{%.2f})'%empord)
    empord, b = np.polyfit(np.log(N_rebalance_list), np.log(np.abs(median_list)), 1)
    ax.scatter(N_rebalance_list, np.abs(median_list), label='median, O(N^{%.2f})'%empord)
    empord, b = np.polyfit(np.log(N_rebalance_list), np.log(variance_list), 1)
    ax.scatter(N_rebalance_list, variance_list, label='variance, O(N^{%.2f})'%empord)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([10 ** (-4), 2 * 10 ** (0)])
    ax.set_xlabel("N")
    plt.legend()
    plt.savefig(delta_hedge_dir + m + "/convergence.png")
    plt.close()

    pnl_metrics[m] = get_pnl_statistics(rel_pnl_T)
dfi.export(pnl_metrics, delta_hedge_dir + '/pnl_metrics.png')


exit(0)
