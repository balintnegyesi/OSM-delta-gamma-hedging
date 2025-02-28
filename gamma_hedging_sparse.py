import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from lib.equations import portfolios as portfolios
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew as scpskew
import dataframe_image as dfi
from scipy import sparse
from multiprocessing import Pool

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


pnl_metrics = pd.DataFrame(columns=model,
                           index=['mean', 'median', 'variance', 'var01', 'var05', 'var95', 'var99', 'es01', 'es05',
                                  'es95', 'es99', 'semivariance', 'skew'])

MODE = 'upper_triangular'
SEC = 'put-exchange'
T_TILDE = 4

N_rebalance_list = [1, 2, 5, 10, 20, 100]

output_dir = './logs/hedging/' + run_name + '/'
delta_hedge_dir = output_dir + "/delta_hedge/"
gamma_hedge_dir = output_dir + '/gamma_hedge/' + MODE + "_" + SEC + "_Ttilde" + str(T_TILDE) + '/'
instruments_dir = output_dir + '/instruments/' + MODE + "_" + SEC + "_Ttilde" + str(T_TILDE) + '/'

with open(output_dir + '/setup.json') as json_data_file:
    setup = json.load(json_data_file)
setup = munch.munchify(setup)
N_finest = setup.N
M = setup.M
chunk_size = setup.chunk_size
number_of_chunks = int(M / chunk_size)

# number_of_chunks = 1
t0_hedge = time.time()
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

        results_dir = gamma_hedge_dir + m + "/N_rebalance=" + str(N_rebalance)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # # # we first need to get the weights
        for chunk_number in range(number_of_chunks):
            print("Starting chunk #%d/%d" % (chunk_number + 1, number_of_chunks))
            dw_chunk = np.load(output_dir + 'dw_sample_chunk_' + str(chunk_number) + ".npy")
            x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")
            exercise_index_chunk = solver.find_exercise_index(x_chunk, N_rebalance)


            inst_dir = instruments_dir + "/chunk_" + str(chunk_number) + "/"


            weights_dir = results_dir + '/chunk_' + str(chunk_number) + "/"
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            np.save(weights_dir + "/exercise_index.npy", exercise_index_chunk)

            for n in range(N_rebalance + 1):
                t0_0 = time.time()
                n_subidx = int(n * mod)
                print("n=%d/%d" % (n, N_rebalance))

                # # # load gamma hedging instruments
                u_n = np.load(inst_dir + "/sec_sample_n_" + str(n_subidx) + ".npy")
                sparse_diff_u_n = sparse.load_npz(inst_dir + "/sparse_dk_sec_sample_n_" + str(n_subidx) + ".npz")
                sparse_hess_u_n = sparse.load_npz(inst_dir + "/sparse_dlk_sec_sample_n_" + str(n_subidx) + ".npz")

                t0 = time.time()
                y_chunk_n, nabla_y_chunk_n, hess_y_chunk_n = solver.gamma_hedging_black_scholes_y_nabla_hess_n(n_subidx, x_chunk[..., n_subidx], is_z_tilde, exercise_index_chunk)
                t1 = time.time()
                time_grad_hess = t1 - t0
                # print("time (grad-hess) %.2f"%(t1-t0))

                # print(hess_y_chunk_n.shape)

                if MODE == 'upper_triangular':
                    num_of_sec = int(solver.bsde.d * (solver.bsde.d + 1) / 2)
                elif MODE == 'full_matrix':
                    num_of_sec = solver.bsde.d ** 2
                elif MODE == 'diagonal':
                    num_of_sec = solver.bsde.d
                else:
                    raise NotImplementedError("Unkown hedging mode")

                def solve_beta_one_path(m):
                    d = int(solver.bsde.d)
                    hess_sec_n_m = sparse_hess_u_n[[m], :].toarray()[0, :].reshape((d, d, num_of_sec))

                    A_m = np.zeros(shape=[num_of_sec, num_of_sec])
                    rhs_m = np.zeros(num_of_sec)

                    if MODE == "full_matrix":
                        try:
                            A_m = hess_sec_n_m.reshape((d ** 2, num_of_sec))
                        except:
                            A_m = hess_sec_n_m.numpy().reshape((d ** 2, num_of_sec))
                        try:
                            rhs_m = hess_y_chunk_n[m, ...].reshape((d ** 2))
                        except:
                            rhs_m = hess_y_chunk_n[m, ...].numpy().reshape((d**2))

                    elif MODE == "upper_triangular":
                        counter = 0
                        for i in range(d):
                            for j in range(i, int(solver.bsde.d)):
                                A_m[counter, :] = hess_sec_n_m[i, j, :]
                                rhs_m[counter] = hess_y_chunk_n[m, i, j]
                                counter += 1

                    elif MODE == "diagonal":
                        counter = 0
                        for i in range(d):
                            A_m[counter, :] = hess_sec_n_m[i, i, :]
                            rhs_m[counter] = hess_y_chunk_n[m, i, i]
                            counter += 1

                    else:
                        raise NotImplementedError


                    return sparse.linalg.lsqr(sparse.csr_array(A_m), rhs_m, btol=1e-8, atol=1e-8)[0]  # atol, btol

                pool = Pool(processes=None)
                t0 = time.time()


                beta = np.array(pool.map(solve_beta_one_path, np.arange(chunk_size)))
                t1 = time.time()
                time_lin_sys = t1 - t0

                t0 = time.time()
                alpha = nabla_y_chunk_n - np.einsum("Mk, Mdk -> Md", beta, sparse_diff_u_n.toarray().reshape(chunk_size, solver.bsde.d, num_of_sec))
                t1 = time.time()
                time_alpha = t1 - t0
                # print("alpha in %.2f"%(t1-t0))

                np.save(weights_dir + "/alpha_h_n_" + str(n) + ".npy", tf.cast(alpha, tf.keras.backend.floatx()))
                np.save(weights_dir + "/beta_h_n_" + str(n) + ".npy", tf.cast(beta, tf.keras.backend.floatx()))
                np.save(weights_dir + "/y_n_" + str(n) + ".npy", tf.cast(y_chunk_n, tf.keras.backend.floatx()))
                t_solve = time.time() - t0_0
                print("execution time: %.2f; grad-hess: %.2f, linear system: %.2f, alpha: %.2f, eta: %.2f" % (t_solve, time_grad_hess, time_lin_sys, time_alpha,  (N_rebalance - n) * t_solve))
        # exit(0)
        # # hedging weights gathered

        # # # gather portfolio
        pnl_gamma_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        relative_pnl_gamma_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        exercise_index = np.zeros(shape=[M, solver.bsde.J], dtype=tf.keras.backend.floatx())

        for chunk_number in range(number_of_chunks):
            print("Starting chunk #%d/%d" % (chunk_number + 1, number_of_chunks))
            dw_chunk = np.load(output_dir + 'dw_sample_chunk_' + str(chunk_number) + ".npy")
            x_chunk = np.load(output_dir + 'x_sample_chunk_' + str(chunk_number) + ".npy")

            weights_dir = results_dir + '/chunk_' + str(chunk_number) + "/"
            exercise_index_chunk = np.load(weights_dir + "/exercise_index.npy")
            sec_dir = instruments_dir + '/chunk_' + str(chunk_number) + "/"
            pnl_gamma_chunk, \
                relative_pnl_gamma_chunk = solver.gamma_hedging_black_scholes_portfolio(x_chunk, sec_dir, weights_dir,
                                                                                        N_rebalance,
                                                                                        exercise_index_chunk)




            idx_low = chunk_number * chunk_size
            idx_high = (chunk_number + 1) * chunk_size

            pnl_gamma_sample[idx_low: idx_high, ...] = pnl_gamma_chunk
            relative_pnl_gamma_sample[idx_low: idx_high, ...] = relative_pnl_gamma_chunk
            exercise_index[idx_low: idx_high, :] = exercise_index_chunk
        results_dir = gamma_hedge_dir + m + "/N_rebalance=" + str(N_rebalance)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.save(results_dir + "/pnl_gamma_sample.npy", pnl_gamma_sample)
        np.save(results_dir + "/relative_pnl_gamma_sample.npy", relative_pnl_gamma_sample)
        np.save(results_dir + "/exercise_index.npy", exercise_index)

        mod = N_finest / N_rebalance
        exercise_date_rebalance = np.ceil(exercise_index / mod)

        rel_pnl_T = np.zeros((M, 1))
        for path_num in range(M):
            rel_pnl_T[path_num, 0] = relative_pnl_gamma_sample[path_num, 0, int(exercise_date_rebalance[path_num, 0])]

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
    plt.savefig(gamma_hedge_dir + m + "/convergence.png")

    pnl_metrics[m] = get_pnl_statistics(rel_pnl_T)
dfi.export(pnl_metrics, gamma_hedge_dir + '/pnl_metrics.png')
t1_hedge = time.time()
print("Total execution time: %.2f seconds"%(t1_hedge-t0_hedge))
exit(0)
