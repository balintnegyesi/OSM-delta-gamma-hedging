import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time, logging
from scipy import sparse
from lib.misc.EUOneDimVanillaCall import EUVanillaCallPut, EUExchangeCall


class SystemReflectedFBSDE(object):
    """Base class for FBSDEs. Notice that an instance of the ForwardProcess class is an attribute to the FBSDE"""

    def __init__(self, eqn_config):
        DTYPE = tf.keras.backend.floatx()
        self.DTYPE = tf.keras.backend.floatx()
        self.eqn_config = eqn_config

        self.J = eqn_config.J
        self.d = eqn_config.d
        self.m = eqn_config.m

        self.T = eqn_config.T
        self.N = eqn_config.N
        self.delta_t = tf.cast(self.T / self.N, dtype=DTYPE).numpy()
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        # self.t = (np.arange(0, self.eqn_config.num_time_interval + 1) * self.delta_t).astype(tf.keras.backend.floatx())
        self.t = np.linspace(0, self.T, self.eqn_config.N + 1).astype(DTYPE)


        self.R_list = self.eqn_config.R
        self._get_reflection_dates(self.eqn_config.R)

        self.is_y_theoretical = False  # for most equations we don't have an analytical solution

        # # # Assets
        self.m = eqn_config.m  # the total number of assets related to the portfolio stacked in one big vector
        # self.rho = eqn_config.parameter_config.rho
        # correl_mtx = self.rho * tf.ones(shape=[self.d, self.d], dtype=tf.keras.backend.floatx())
        # self.correl_mtx = tf.linalg.set_diag(correl_mtx, tf.ones(self.d, dtype=tf.keras.backend.floatx()))
        # self.cholesky = tf.linalg.cholesky(self.correl_mtx)

    def _get_reflection_dates(self, R_list):
        '''
        R_list is a J-long list which contains the number of (equidistant) exercise intervals
        from that this function updates the list self.reflection_dates to a list of lists, such that we allow for
        separate exercising
        '''
        if len(R_list) != self.J:
            raise ValueError("R_list should be of length J=%d, got %d instead"%(self.J, len(R_list)))


        reflection_dates = []
        reflection_subsequence_indices = []  # list of lists; each element correspdonds to the list of indices corresponding to the
        for j in range(self.J):
            R_j = R_list[j]
            if self.N % R_j != 0:
                return ValueError("R_list contains non subset monitoring grid")
            factor_j = int(self.N / R_j)
            reflection_subsequence_indices.append(np.arange(0, R_j + 1) * factor_j)
            delta_r_j = self.T / R_j
            set_of_dates_j = np.linspace(delta_r_j, self.T - delta_r_j, R_j - 1).astype(self.DTYPE)
            reflection_dates.append(set_of_dates_j)

            if len(reflection_subsequence_indices[j]) != (R_j + 1):
                raise ValueError

            if len(reflection_dates[j]) != (R_j - 1):
                raise ValueError


        is_exercise_date = np.zeros([self.J, self.N + 1]).astype(self.DTYPE)
        for n in range(1, self.N):
            for j in range(self.J):
                is_exercise_date[j, n] = np.where(n in reflection_subsequence_indices[j], 1.0, 0.0)


        self.reflection_dates = reflection_dates
        self.reflection_sub_idx = reflection_subsequence_indices
        self.is_exercise_date = tf.cast(is_exercise_date, dtype=self.DTYPE)

        return 0

    def _get_cholesky(self, correlation_matrix):
        self.correlation_mtx = correlation_matrix
        self.cholesky = tf.linalg.cholesky(self.correlation_mtx)
        self.inv_cholesky = tf.linalg.pinv(self.cholesky)

        return 0
    def mu_process_tf(self, t, x):
        raise NotImplementedError

    def sigma_process_tf(self, t, x):
        raise NotImplementedError
    @tf.function
    def inverse_sigma_process_tf(self, t, x):
        return tf.linalg.pinv(self.sigma_process_tf(t, x))

    def inverse_sigma_transpose_process_tf(self, t, x):
        return tf.linalg.pinv(tf.transpose(self.sigma_process_tf(t, x), perm=[0, 2, 1]))

    @tf.function(reduce_retracing=True)
    def nabla_mu_process_tf(self, t, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            mu = self.mu_process_tf(t, x)
        return tape.batch_jacobian(mu, x)

    @tf.function(reduce_retracing=True)
    def nabla_sigma_process_tf(self, t, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            sigma = self.sigma_process_tf(t, x)
        return tape.batch_jacobian(sigma, x)

    @tf.function(reduce_retracing=True)
    def nabla_sigma_process_jth_col_tf(self, t, x, j):
        with tf.GradientTape() as tape:
            tape.watch(x)
            sigma_jth_row = self.sigma_process_tf(t, x)[..., j]
        return tape.batch_jacobian(sigma_jth_row, x)

    @tf.function(reduce_retracing=True)
    def nabla_sigma_process_ith_row_tf(self, t, x, i):
        with tf.GradientTape() as tape:
            tape.watch(x)
            sigma_ith_row = self.sigma_process_tf(t, x)[:, i, :]
        return tape.batch_jacobian(sigma_ith_row, x)

    @tf.function(reduce_retracing=True)
    def sample(self, num_sample):
        raise NotImplementedError

    @tf.function(reduce_retracing=True)
    def sample_step_malliavin_euler_tf(self, dw_sample, x_sample, n):
        """
        receives a bunch of simulations; and an index 'n' in the discrete time grid
        returns the 'n+1'th time step's malliavin derivative calculated by an Euler-Maruyama scheme
        :param dw_sample:
        :param x_sample:
        :param n:
        :return:
        """
        t_n = self.t[n]
        dw_n = dw_sample[:, :, n]
        x_n = x_sample[:, :, n]
        sigma_n = self.sigma_process_tf(t_n, x_n)
        DnXn = 1 * sigma_n
        nabla_mu_n = self.nabla_mu_process_tf(t_n, x_n)
        time_integral = self.delta_t * tf.einsum('Mij, Mjk -> Mik', nabla_mu_n, DnXn)

        # # # let's calculate the Ito part via summing over axes, so that we do not have to construct an Mxdxdxd tensor
        # # # for nabla_x sigma
        ito_integral = tf.zeros(tf.shape(time_integral), dtype=tf.keras.backend.floatx())
        for j in range(self.d):
            nabla_sigma_jth_row = self.nabla_sigma_process_jth_col_tf(t_n, x_n, j)
            dw_n_jth_row = dw_n[:, j]
            ito_integral += tf.einsum("Mij, M -> Mij",
                                      tf.einsum("Mij, Mjk -> Mik", nabla_sigma_jth_row, DnXn),
                                      dw_n_jth_row)

        DnXnext = DnXn + time_integral + ito_integral

        return DnXnext

    @tf.function
    def sample_step_malliavin(self, dw_sample, x_sample, n):
        return self.sample_step_malliavin_euler_tf(dw_sample, x_sample, n)

    def f_tf(self, t, x, y, z):
        """ Generator function in the system of PDEs. """
        raise NotImplementedError

    def g_tf(self, t, x):
        """ Terminal condition of the system of PDEs. """
        raise NotImplementedError

    @tf.function(reduce_retracing=True)
    def gradx_g_tf(self, t, x):
        """
        just like any partial differentiation, this will default to automatic differentation.
        if the user is not lazy: these can be implemented as tf calculations for each equation separately (cheaper)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            g = self.g_tf(t, x)
        return tape.batch_jacobian(g, x)

    @tf.function(reduce_retracing=True)
    def gradx_f_tf(self, t, x, y, z):
        with tf.GradientTape() as tape:
            tape.watch(x)
            f = self.f_tf(t, x, y, z)
        gradx_f = tape.batch_jacobian(f, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return gradx_f

    @tf.function(reduce_retracing=True)
    def grady_f_tf(self, t, x, y, z):
        with tf.GradientTape() as tape:
            tape.watch(y)
            f = self.f_tf(t, x, y, z)
        return tape.batch_jacobian(f, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @tf.function(reduce_retracing=True)
    def gradz_f_tf(self, t, x, y, z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            f = self.f_tf(t, x, y, z)
        return tape.batch_jacobian(f, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    @tf.function(reduce_retracing=True)
    def hessx_g_tf(self, t, x_tf):
        with tf.GradientTape() as tape1:
            tape1.watch(x_tf)
            with tf.GradientTape() as tape2:
                tape2.watch(x_tf)
                g = self.g_tf(t, x_tf)
            grad_g = tape2.batch_jacobian(g, x_tf)
        hess_g = tape1.batch_jacobian(grad_g, x_tf)
        return hess_g

    def get_gradient_from_z(self, t, x, z, is_held_at_t):
        # # # is_held_at_t is a tensor of shape M x J valued 1.0 if the option has not been exercised at any s<t, and 0.0 otherwise
        return tf.einsum("Md, Mdm -> Mm", tf.einsum("MJ, MJd -> Md", is_held_at_t, z), self.inverse_sigma_process_tf(t, x))

    def get_hessian_from_gamma(self, t, x, grad_u, gamma, is_held_at_t):
        # # # is_held_at_t is a tensor of shape M x J valued 1.0 if the option has not been exercised at any s<t, and 0.0 otherwise
        # # # gamma = (hess-y^T x sigma)^T + grad-y x grad-sigma

        nabla_u_nabla_sigma = tf.einsum("M, Mqm -> Mqm", grad_u[..., 0], self.nabla_sigma_process_ith_row_tf(t, tf.constant(x), 0))
        for i in range(self.d):
            nabla_u_nabla_sigma += tf.einsum("M, Mqm -> Mqm", grad_u[..., i], self.nabla_sigma_process_ith_row_tf(t, tf.constant(x), i))

        hess = tf.einsum("Mij, Mik -> Mjk",
                         self.inverse_sigma_process_tf(t, x),
                         tf.einsum("MJ, MJik -> Mik", is_held_at_t, gamma) - nabla_u_nabla_sigma)

        return hess

    @staticmethod
    def exchange_call_option(t, S_k, S_j, K_jk, T, sigma_k, sigma_j, rho, q_k, q_j):
        '''
        to price a European exchange call with the payoff (S_T^k - K^{kj} S_T^j)^+
        with Black Scholes dynamics
        by the Margrabe formula

        update: each asset is allowed to pay out a constant continuous dividend with rate q_k, q_j
        '''
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho * sigma_k * sigma_j)
        d_1 = 1 / (sigma_jk * np.sqrt(T - t)) * (np.log(S_k / (K_jk * S_j)) + (q_j - q_k + sigma_jk ** 2 / 2) * (T - t))
        d_2 = d_1 - sigma_jk * np.sqrt(T - t)
        normal_dist = tf.compat.v1.distributions.Normal(loc=0.0, scale=1.0)

        # # # price
        Phi_d1 = tf.cast(normal_dist.cdf(d_1), dtype=tf.keras.backend.floatx())  # std normal CDFs
        Phi_d2 = tf.cast(normal_dist.cdf(d_2), dtype=tf.keras.backend.floatx())
        C_jk = np.exp(-q_k * (T - t)) * S_k * Phi_d1 - np.exp(-q_j * (T - t)) * K_jk * S_j * Phi_d2

        # # # derivative
        dj_C_jk = -np.exp(-q_j * (T - t)) * K_jk * Phi_d2
        dk_C_jk = np.exp(-q_k * (T - t)) * Phi_d1

        # # # second-order derivatives
        phi_d1 = tf.cast(normal_dist.prob(d_1), dtype=tf.keras.backend.floatx())  # std normal PDFs
        phi_d2 = tf.cast(normal_dist.prob(d_2), dtype=tf.keras.backend.floatx())  # std normal PDFs
        normalization = sigma_jk * np.sqrt(T - t)

        djj_C_jk = np.exp(-q_j * (T - t)) * K_jk * phi_d2 / S_j / normalization
        djk_C_jk = -np.exp(-q_j * (T - t)) * phi_d1 / S_j / normalization
        dkj_C_jk = 1 * djk_C_jk
        dkk_C_jk = np.exp(-q_k * (T - t)) * phi_d1 / S_k / normalization

        return C_jk, dj_C_jk, dk_C_jk, djj_C_jk, djk_C_jk, dkj_C_jk, dkk_C_jk

    def get_hedging_securities(self, x_sample, mode, securities='call-exchange', K_ex=None, K_1d=None, T_tilde=None):
        M, d, Np1 = x_sample.shape

        if mode == 'upper_triangular':
            num_of_sec = int(self.d * (self.d + 1) / 2)
        elif mode == 'full_matrix':
            num_of_sec = self.d ** 2
        elif mode == 'diagonal':
            num_of_sec = self.d
        else:
            raise NotImplementedError("Unkown hedging mode")

        if securities == 'call-exchange':
            EUCall = EUVanillaCallPut('call')
            xchange_call = EUExchangeCall()
        elif securities == 'put-exchange':
            EUCall = EUVanillaCallPut('put')
            xchange_call = EUExchangeCall()
        else:
            raise NotImplementedError("Unknown hedging securities")

        # if K is None:
        #     K = self.K
        # if T_tilde is None:
        #     T_tilde = 2 * self.total_time

        I_ij = np.zeros(shape=[M, num_of_sec, Np1], dtype=tf.keras.backend.floatx())
        dk_I_ij = np.zeros(shape=[M, d, num_of_sec, Np1], dtype=tf.keras.backend.floatx())
        dlk_I_ij = np.zeros(shape=[M, d, d, num_of_sec, Np1], dtype=tf.keras.backend.floatx())

        if mode == 'full_matrix':
            # # # d^2-many securities
            for n in range(Np1):
                t0 = time.time()
                t_n = self.t[n]
                x_n = x_sample[..., n]

                sec_idx = 0  # security index
                for j in range(d):
                    for k in range(d):
                        S_k = x_n[:, k]
                        sigma_k = self.sigma[k]
                        rho_jk = self.correlation_mtx[j, k]
                        q_k = self.q
                        q_j = self.q
                        if j == k:
                            # # # should be vanilla call
                            S_j = np.exp(-self.r * (T_tilde - t_n)) * np.ones(S_k.shape)

                            C_jk = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                            dk_C_jk = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                            dkk_C_jk = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                        else:
                            S_j = x_n[:, j]
                            sigma_j = self.sigma[j]

                            (C_jk, dj_C_jk, dk_C_jk,
                             djj_C_jk, djk_C_jk, dkj_C_jk, dkk_C_jk) = self.exchange_call_option(t_n, S_k, S_j, K_ex,
                                                                                                 T_tilde,
                                                                                                 sigma_k, sigma_j,
                                                                                                 rho_jk, q_k, q_j)

                        I_ij[:, sec_idx, n] = C_jk
                        dk_I_ij[:, k, sec_idx, n] = dk_C_jk
                        dlk_I_ij[:, k, k, sec_idx, n] = dkk_C_jk
                        if j != k:
                            dk_I_ij[:, j, sec_idx, n] = dj_C_jk
                            dlk_I_ij[:, j, j, sec_idx, n] = djj_C_jk
                            dlk_I_ij[:, k, j, sec_idx, n] = djk_C_jk
                            dlk_I_ij[:, j, k, sec_idx, n] = dkj_C_jk

                        sec_idx += 1

                t1 = time.time()
                print("hedging securities: n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))
        elif mode == 'upper_triangular':
            # # # d * (d + 1) / 2-many securities
            for n in range(Np1):
                t0 = time.time()
                t_n = self.t[n]
                x_n = x_sample[..., n]

                sec_idx = 0  # security index
                for j in range(d):
                    for k in range(j, d):
                        S_k = x_n[:, k]
                        sigma_k = self.sigma[k]
                        rho_jk = self.correlation_mtx[j, k]
                        q_k = self.q
                        q_j = self.q
                        if j == k:
                            # # # should be vanilla call
                            C_jk = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                            dk_C_jk = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                            dkk_C_jk = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                        else:
                            S_j = x_n[:, j]
                            sigma_j = self.sigma[j]

                            (C_jk, dj_C_jk, dk_C_jk,
                             djj_C_jk, djk_C_jk, dkj_C_jk, dkk_C_jk) = self.exchange_call_option(t_n, S_k, S_j, K_ex,
                                                                                                 T_tilde,
                                                                                                 sigma_k, sigma_j,
                                                                                                 rho_jk, q_k, q_j)

                        I_ij[:, sec_idx, n] = C_jk
                        dk_I_ij[:, k, sec_idx, n] = dk_C_jk
                        dlk_I_ij[:, k, k, sec_idx, n] = dkk_C_jk
                        if j != k:
                            dk_I_ij[:, j, sec_idx, n] = dj_C_jk
                            dlk_I_ij[:, j, j, sec_idx, n] = djj_C_jk
                            dlk_I_ij[:, k, j, sec_idx, n] = djk_C_jk
                            dlk_I_ij[:, j, k, sec_idx, n] = dkj_C_jk

                        sec_idx += 1
                t1 = time.time()
                print("n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))

        elif mode == 'diagonal':
            # # # d-many securities
            for n in range(Np1):
                t0 = time.time()
                t_n = self.t[n]
                x_n = x_sample[..., n]

                for k in range(d):
                    S_k = x_n[:, k]
                    sigma_k = self.sigma[k]
                    q_k = self.q
                    C_k = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                    dk_C_k = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                    dkk_C_k = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                    I_ij[:, k, n] = C_k
                    dk_I_ij[:, k, k, n] = dk_C_k
                    dlk_I_ij[:, k, k, k, n] = dkk_C_k
                t1 = time.time()
                # print("n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))

        return I_ij, dk_I_ij, dlk_I_ij

    def get_hedging_securities_at_t_n_sparse(self, n, x_sample, mode, securities='call-exchange',
                                             K_ex=None, K_1d=None, T_tilde=None):
        M, d, Np1 = x_sample.shape

        if mode == 'upper_triangular':
            num_of_sec = int(self.d * (self.d + 1) / 2)
        elif mode == 'full_matrix':
            num_of_sec = self.d ** 2
        elif mode == 'diagonal':
            num_of_sec = self.d
        else:
            raise NotImplementedError("Unkown hedging mode")

        if securities == 'call-exchange':
            EUCall = EUVanillaCallPut('call')
            xchange_call = EUExchangeCall()
        elif securities == 'put-exchange':
            EUCall = EUVanillaCallPut('put')
            xchange_call = EUExchangeCall()
        else:
            raise NotImplementedError("Unknown hedging securities")

        # if K is None:
        #     K = self.K
        # if T_tilde is None:
        #     T_tilde = 2 * self.total_time


        I_ij_n = np.zeros(shape=[M, num_of_sec], dtype=tf.keras.backend.floatx())
        dk_I_ij_n = sparse.lil_matrix((M, d * num_of_sec), dtype=tf.keras.backend.floatx())
        dlk_I_ij_n = sparse.lil_matrix((M, d**2 * num_of_sec), dtype=tf.keras.backend.floatx())

        if mode == 'full_matrix':
            # raise NotImplementedError
            # # # d^2-many securities
            t0 = time.time()
            t_n = self.t[n]
            x_n = x_sample[..., n]

            sec_idx = 0  # security index
            for j in range(d):
                for k in range(d):
                    S_k = x_n[:, k]
                    sigma_k = self.sigma[k]
                    rho_jk = self.correlation_mtx[j, k]
                    q_k = self.q
                    q_j = self.q
                    if j == k:
                        # # # should be vanilla call
                        S_j = np.exp(-self.r * (T_tilde - t_n)) * np.ones(S_k.shape)

                        C_jk = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                        dk_C_jk = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                        dkk_C_jk = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                    else:
                        S_j = x_n[:, j]
                        sigma_j = self.sigma[j]

                        (C_jk, dj_C_jk, dk_C_jk,
                         djj_C_jk, djk_C_jk, dkj_C_jk, dkk_C_jk) = self.exchange_call_option(t_n, S_k, S_j, K_ex,
                                                                                             T_tilde,
                                                                                             sigma_k, sigma_j,
                                                                                             rho_jk, q_k, q_j)

                    I_ij_n[:, sec_idx] = C_jk
                    dk_I_ij_n[:, k * num_of_sec + sec_idx] = dk_C_jk
                    dlk_I_ij_n[:, self.get_sparse_index(k, k, sec_idx, d, d, num_of_sec)] = dkk_C_jk

                    if j != k:
                        dk_I_ij_n[:, j * num_of_sec + sec_idx] = dj_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(j, j, sec_idx, d, d, num_of_sec)] = djj_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(k, j, sec_idx, d, d, num_of_sec)] = djk_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(j, k, sec_idx, d, d, num_of_sec)] = dkj_C_jk

            t1 = time.time()
            print("hedging securities: n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))
        elif mode == 'upper_triangular':
            # # # d * (d + 1) / 2-many securities
            t0 = time.time()
            t_n = self.t[n]
            x_n = x_sample[..., n]

            sec_idx = 0  # security index
            for j in range(d):
                for k in range(j, d):
                    S_k = x_n[:, k]
                    sigma_k = self.sigma[k]
                    rho_jk = self.correlation_mtx[j, k]
                    q_k = self.q
                    q_j = self.q
                    if j == k:
                        # # # should be vanilla call
                        C_jk = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                        dk_C_jk = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                        dkk_C_jk = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                    else:
                        S_j = x_n[:, j]
                        sigma_j = self.sigma[j]

                        (C_jk, dj_C_jk, dk_C_jk,
                         djj_C_jk, djk_C_jk, dkj_C_jk, dkk_C_jk) = self.exchange_call_option(t_n, S_k, S_j, K_ex,
                                                                                             T_tilde, sigma_k, sigma_j,
                                                                                             rho_jk, q_k, q_j)
                    # I_ij_n[:, sec_idx] = C_jk
                    # dk_I_ij_n[:, k, sec_idx] = dk_C_jk
                    # dlk_I_ij_n[:, k, k, sec_idx] = dkk_C_jk
                    # if j != k:
                    #     dk_I_ij_n[:, j, sec_idx] = dj_C_jk
                    #     dlk_I_ij_n[:, j, j, sec_idx] = djj_C_jk
                    #     dlk_I_ij_n[:, k, j, sec_idx] = djk_C_jk
                    #     dlk_I_ij_n[:, j, k, sec_idx] = dkj_C_jk


                    I_ij_n[:, sec_idx] = C_jk
                    dk_I_ij_n[:, k * num_of_sec + sec_idx] = dk_C_jk
                    dlk_I_ij_n[:, self.get_sparse_index(k, k, sec_idx, d, d, num_of_sec)] = dkk_C_jk

                    if j != k:
                        dk_I_ij_n[:, j * num_of_sec + sec_idx] = dj_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(j, j, sec_idx, d, d, num_of_sec)] = djj_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(k, j, sec_idx, d, d, num_of_sec)] = djk_C_jk
                        dlk_I_ij_n[:, self.get_sparse_index(j, k, sec_idx, d, d, num_of_sec)] = dkj_C_jk

                    sec_idx += 1
            t1 = time.time()
            print("n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))

        elif mode == 'diagonal':
            # raise NotImplementedError
            # # # d-many securities
            t0 = time.time()
            t_n = self.t[n]
            x_n = x_sample[..., n]

            for k in range(d):
                S_k = x_n[:, k]
                sigma_k = self.sigma[k]
                q_k = self.q
                C_k = EUCall.price(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                dk_C_k = EUCall.delta(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)
                dkk_C_k = EUCall.gamma(S_k, K_1d, self.r, sigma_k, T_tilde - t_n, q_k)

                # I_ij_n[:, k] = C_k
                # dk_I_ij_n[:, k, k] = dk_C_k
                # dlk_I_ij_n[:, k, k, k] = dkk_C_k

                I_ij_n[:, k] = C_k
                dk_I_ij_n[:, k * num_of_sec + k] = dk_C_k
                dlk_I_ij_n[:, self.get_sparse_index(k, k, k, d, d, num_of_sec)] = dkk_C_k
            t1 = time.time()
            # print("n=%d, time=%.1f, ETA=%.1f" % (n, (t1 - t0), (Np1 - n) * (t1 - t0)))

        return I_ij_n, dk_I_ij_n, dlk_I_ij_n
    @staticmethod
    def get_sparse_index(d, q, k, D, Q, K):
        # input index triple (d, q, k) with dimension shape (D, Q, K)
        # returns the corresponding "C" type reshape one-dim index 0..D*Q*K
        return d * (Q * K) + q * K + k

    def LSE_alpha_smoothing(self, x, axis=-1, is_rescaled=True, x_star=None):
        # # # approximates max_i x_i with LSE smoothing
        if is_rescaled:
            if x_star is None:
                # take default x_star
                x_star = tf.math.reduce_max(x, axis=axis, keepdims=True)
                # x_star = tf.tile(tf.math.reduce_max(x, axis=axis, keepdims=True), multiples=[1, tf.shape(x)[-1]])
            else:
                x_star = x_star * 1
        else:
            x_star = 0

        # LSE formula: 1/alpha * log(sum_i exp(alpha * (x_i - x_star))) + x_star

        argument = tf.reduce_sum(tf.math.exp(self.LSE_alpha * (x - x_star)), axis=axis, keepdims=True)

        LSE_alpha_star = 1 / self.LSE_alpha * tf.math.log(argument) + x_star

        return LSE_alpha_star


    def reflection_operator(self, t, x, y_c, is_smoothed=False):
        # y_c refers to the continuation value
        if t in self.reflection_dates:
            payoff = self.g_tf(t, x)
            if is_smoothed:
                arg_max = tf.concat([payoff - y_c, tf.zeros(payoff.shape, dtype=tf.keras.backend.floatx())], axis=-1)
                reflection_term = self.LSE_alpha_smoothing(arg_max, axis=-1, is_rescaled=True, x_star=None)
            else:
                reflection_term = tf.where(payoff - y_c > 0, payoff - y_c, 0)
        else:
            reflection_term = tf.zeros(tf.shape(y_c), dtype=tf.keras.backend.floatx())
        return y_c + reflection_term


class HestonBaseClass(SystemReflectedFBSDE):
    """
    assuming that the assets are all Heston driven
    """
    def __init__(self, eqn_config):
        super(HestonBaseClass, self).__init__(eqn_config)

        if self.m != self.d / 2:
            raise ValueError

        self.mu = self.eqn_config.parameter_config.mu
        self.kappa = self.eqn_config.parameter_config.kappa
        self.nu_bar = self.eqn_config.parameter_config.nu_bar
        self.xi = self.eqn_config.parameter_config.xi
        self.X_0 = self.eqn_config.parameter_config.X_0
        self.nu_0 = self.eqn_config.parameter_config.nu_0
        if hasattr(eqn_config.parameter_config, "rho"):
            self.correlation_parameter = eqn_config.parameter_config.rho  # assumed to be constant for each pair
        else:
            raise ValueError("Correlated Heston requested, without correlation parameter passed")
        self.non_negativity_approach = self.eqn_config.parameter_config.non_negativity_approach
        self.epsilon = self.eqn_config.parameter_config.epsilon

        # self.d_bar = 4 * self.kappa * self.nu_bar / (self.xi ** 2)

        correl_mtx = self.correlation_parameter * tf.ones(shape=[self.d, self.d], dtype=tf.keras.backend.floatx())
        correl_mtx = tf.linalg.set_diag(correl_mtx, tf.ones(self.d, dtype=tf.keras.backend.floatx()))

        # self.cholesky = tf.linalg.cholesky(correl_mtx)  # this is the lower triangular version by default
        self._get_cholesky(correl_mtx)

# =============================================================================
    # def nabla_sigma_process_jth_col_tf(self, t, x, j):
    #     # sigma should be the covariance matrix:
    #
    #     M = tf.shape(x)[0]
    #     d = tf.shape(x)[1]
    #
    #     stock_price = x[:, 0: self.m]
    #     nu = x[:, self.m:]
    #
    #     basis_jth = tf.one_hot([j], self.m, dtype=tf.keras.backend.floatx())
    #     batched_basis_jth = tf.repeat(basis_jth, tf.shape(x)[0], axis=0)
    #     if j < self.m:
    #         # wrt a stock
    #         stock_diag_part = tf.pow(nu, 0.5) * batched_basis_jth
    #         vol_diag_part = 0 * nu
    #     else:
    #         # wrt a volatility
    #         stock_diag_part = 0 * stock_price
    #         vol_diag_part = 0.5 * tf.pow(nu, -0.5) * (self.xi + stock_price) * batched_basis_jth
    #
    #     diag_part = tf.concat([stock_diag_part, vol_diag_part], axis=-1)  # M x (2d)
    #
    #     correlated_nablaj_sigma = tf.linalg.diag(diag_part)  # this will still have to be multiplied with the Cholesky
    #
    #     uncorrelated_nablaj_sigma = tf.einsum("Mij, jk -> Mik", correlated_nablaj_sigma, self.cholesky)
    #
    #     return uncorrelated_nablaj_sigma
#     @tf.function
#     def sample_step_malliavin(self, dw_sample, x_sample, n):
#         """
#         receives a bunch of simulations; and an index 'n' in the discrete time grid
#         returns the 'n+1'th time step's malliavin derivative calculated by an Euler-Maruyama scheme
#         :param dw_sample:
#         :param x_sample:
#         :param n:
#         :return:
#         """
#         t_n = self.t[n]
#         dw_n = dw_sample[:, :, n]
#         x_n = x_sample[:, :, n]
#         sigma_n = self.sigma_process_tf(t_n, x_n)
#         DnXn = 1 * sigma_n
#         nabla_mu_n = self.nabla_mu_process_tf(t_n, x_n)
#         time_integral = self.delta_t * tf.einsum('Mij, Mjk -> Mik', nabla_mu_n, DnXn)
#         # # # let's calculate the Ito part via summing over axes, so that we do not have to construct an Mxdxdxd tensor
#         # # # for nabla_x sigma
#         vec_n = tf.einsum("Mij, Mj -> Mi", DnXn, dw_n)
#         # condition = lambda j, output: self.ito_condition(j)
#         # body = lambda j, output: self.ito_body(j, output, t_n, x_n, vec_n)
#         # output = tf.zeros(shape=[tf.shape(time_integral)[0], tf.shape(time_integral)[1], 1])
#         # j, ito_part = tf.while_loop(condition, body, [tf.constant(0), output], parallel_iterations=self.dim)
#         # ito_integral = ito_part[:, :, 1:]
#         ito_integral = tf.zeros(tf.shape(time_integral), dtype=tf.keras.backend.floatx())
#         for j in range(self.d):
#             nabla_sigma_jth_row = self.nabla_sigma_process_jth_col_tf(t_n, x_n, j)
#             dw_n_jth_row = dw_n[:, j]
#             ito_integral += tf.einsum("Mijk, Mk -> Mij",
#                                       tf.expand_dims(tf.einsum("Mij, Mjk -> Mik", nabla_sigma_jth_row, DnXn), -1),
#                                       tf.expand_dims(dw_n_jth_row, -1))
# 
#         DnXnext = DnXn + time_integral + ito_integral
# 
#         return tf.convert_to_tensor(DnXnext)
# =============================================================================

    def mu_process_tf(self, t, x):
        # first d dimension of x corresponds to stock price, second d to the variance process
        stock_price = x[:, 0: self.m]
        nu = self.f_2_truncation(x[:, self.m:])

        stock_drift = self.mu * stock_price  # slicing is closed, open
        vol_drift = self.kappa * (self.nu_bar - nu)

        return tf.concat([stock_drift, vol_drift], axis=-1)

    def sigma_process_tf(self, t, x):
        # sigma should be the covariance matrix:
        stock_price = x[:, 0: self.m]
        nu = self.f_3_truncation(x[:, self.m:])

        stock_diagonal = stock_price * tf.pow(nu, 0.5)
        vol_diagonal = self.xi * tf.pow(nu, 0.5)

        diagonal = tf.concat([stock_diagonal, vol_diagonal], axis=-1)  # M x (2d)
        correlated_sigma = tf.linalg.diag(diagonal)

        uncorrelated_sigma = tf.einsum("Mij, jk -> Mik", correlated_sigma, self.cholesky)

        return uncorrelated_sigma

    def inverse_sigma_process_tf(self, t, x):
        # sigma should be the covariance matrix:
        stock_price = x[:, 0: self.m]
        nu = self.f_3_truncation(x[:, self.m:])

        stock_diagonal = stock_price * tf.pow(nu, 0.5)
        vol_diagonal = self.xi * tf.pow(nu, 0.5)

        diagonal = tf.concat([stock_diagonal, vol_diagonal], axis=-1)  # M x (2d)
        inverse_diagonal_part = tf.linalg.diag(tf.pow(diagonal, -1))

        return tf.einsum("jk, Mki -> Mji", self.inv_cholesky, inverse_diagonal_part)
    def c_bar(self, dt):
        return self.xi ** 2 * (1 - tf.cast(tf.math.exp(-self.kappa * dt), self.DTYPE)) / (4 * self.kappa)
    def lambda_bar(self, dt, t):
        return tf.cast(tf.math.exp(-self.kappa * dt), self.DTYPE) / self.c_bar(dt)
    def f_1_truncation(self, x):
        if self.non_negativity_approach == 'absorption':
            retval = tf.where(x >= 0, x, 0)
        elif self.non_negativity_approach == 'reflection':
            retval = tf.abs(x)
        else:
            retval = x
        return retval

    def f_2_truncation(self, x):
        if self.non_negativity_approach == 'absorption':
            retval = tf.where(x >= 0, x, 0)
        elif self.non_negativity_approach == 'reflection':
            retval = tf.abs(x)
        elif self.non_negativity_approach == 'full_truncation':
            retval = tf.where(x >= 0, x, 0)
        else:
            retval = x

        return retval

    def f_3_truncation(self, x):
        if self.non_negativity_approach == 'absorption':
            retval = tf.where(x >= 0, x, 0)
        elif self.non_negativity_approach == 'reflection':
            retval = tf.abs(x)
        elif self.non_negativity_approach == 'higham-mao':
            retval = tf.abs(x)
        elif self.non_negativity_approach == 'partial_truncation':
            retval = tf.where(x >= 0, x, 0)
        elif self.non_negativity_approach == "full_truncation":
            retval = tf.where(x >= 0, x, 0)
        return tf.math.maximum(retval, self.epsilon)

    def f_4_truncation(self, x):
        return self.f_2_truncation(x)

    def f_5_truncation(self, x):
        return self.f_3_truncation(x)

    @tf.function
    def sample_new(self, num_sample, dw_sample=None):
        # return self.log_euler_sampling(num_sample)
        # Euler-Maruyama
        if dw_sample is None:
            dw_sample = tf.random.normal(shape=[num_sample, self.d, self.N],
                                         dtype=tf.keras.backend.floatx()) * np.sqrt(self.delta_t)
        else:
            num_sample = dw_sample.shape[0]
        x_0 = self.X_0 * tf.ones(shape=[num_sample, self.m], dtype=tf.keras.backend.floatx())
        nu_0 = self.nu_0 * tf.ones(shape=[num_sample, self.m], dtype=tf.keras.backend.floatx())

        tf.stop_gradient(dw_sample)

        x_n = tf.concat([x_0, nu_0], axis=-1)

        x_sample = x_n[..., None]

        for n in range(self.N):
            t_n = self.t[n]
            dw_n = dw_sample[:, :, n]

            s_n = x_n[:, 0: self.m]
            nu_n = x_n[:, self.m:]

            mu_n = self.mu_process_tf(t_n, tf.concat([s_n, self.f_2_truncation(nu_n)], axis=-1))
            sigma_n = self.sigma_process_tf(t_n, tf.concat([s_n, self.f_3_truncation(nu_n)], axis=-1))

            x_n = self.f_3_truncation(tf.concat([s_n, self.f_1_truncation(nu_n)], axis=-1)
                                      + mu_n * self.delta_t
                                      + tf.einsum("Mij, Mj -> Mi", sigma_n, dw_n))

            x_sample = tf.concat([x_sample, tf.expand_dims(x_n, -1)], axis=-1)

        return dw_sample, x_sample

    @tf.function
    def sample(self, num_sample, dw_sample=None):
        # return self.log_euler_sampling(num_sample)
        # Euler-Maruyama
        if dw_sample is None:
            dw_sample = tf.random.normal(shape=[num_sample, self.d, self.N],
                                         dtype=tf.keras.backend.floatx()) * np.sqrt(self.delta_t)
        else:
            num_sample = dw_sample.shape[0]
        x_0 = self.X_0 * tf.ones(shape=[num_sample, self.m], dtype=tf.keras.backend.floatx())
        nu_0 = self.nu_0 * tf.ones(shape=[num_sample, self.m], dtype=tf.keras.backend.floatx())



        tf.stop_gradient(dw_sample)

        x_n = tf.concat([x_0, nu_0], axis=-1)

        x_sample = x_n[..., None]

        for n in range(self.N):
            t_n = self.t[n]
            dw_n = dw_sample[:, :, n]

            s_n = x_n[:, 0: self.m]
            nu_n = x_n[:, self.m:]


            mu_n = self.mu_process_tf(t_n, tf.concat([s_n, self.f_2_truncation(nu_n)], axis=-1))
            sigma_n = self.sigma_process_tf(t_n, tf.concat([s_n, self.f_3_truncation(nu_n)], axis=-1))

            x_n = self.f_3_truncation(tf.concat([s_n, self.f_1_truncation(nu_n)], axis=-1)
                                      + mu_n * self.delta_t
                                      + tf.einsum("Mij, Mj -> Mi", sigma_n, dw_n))


            x_sample = tf.concat([x_sample, tf.expand_dims(x_n, -1)], axis=-1)

        return dw_sample, x_sample

class HestonLogAssetClass(HestonBaseClass):
    def __init__(self, eqn_config):
        super(HestonLogAssetClass, self).__init__(eqn_config)

        self.X_0 = np.log(self.X_0).astype(tf.keras.backend.floatx())  # log transform is needed
        print(self.X_0)

    # def nabla_sigma_process_jth_col_tf(self, t, x, j):
    #     # sigma should be the covariance matrix:
    #
    #     M = tf.shape(x)[0]
    #     d = tf.shape(x)[1]
    #
    #     stock_price = x[:, 0: self.m]
    #     nu = x[:, self.m:]
    #     # if self.non_negativity_approach == 'abs':
    #     #     nu = tf.abs(nu)
    #     # elif self.non_negativity_approach == 'floor':
    #     #     nu = tf.where(nu >= 0, nu, self.epsilon)
    #     # else:
    #     #     raise NotImplementedError
    #
    #     basis_jth = tf.one_hot([j], self.m, dtype=tf.keras.backend.floatx())
    #     batched_basis_jth = tf.repeat(basis_jth, tf.shape(x)[0], axis=0)
    #     if j < self.m:
    #         # wrt a stock
    #         stock_diag_part = 0.0 * stock_price
    #         vol_diag_part = 0 * nu
    #     else:
    #         # wrt a volatility
    #         stock_diag_part = 0 * stock_price
    #         vol_diag_part = 0.5 * tf.pow(nu, -0.5) * (self.xi + 1.0) * batched_basis_jth
    #
    #     diag_part = tf.concat([stock_diag_part, vol_diag_part], axis=-1)  # M x (2d)
    #
    #     correlated_nablaj_sigma = tf.linalg.diag(diag_part)  # this will still have to be multiplied with the Cholesky
    #
    #     uncorrelated_nablaj_sigma = tf.einsum("Mij, jk -> Mik", correlated_nablaj_sigma, self.cholesky)
    #
    #     return uncorrelated_nablaj_sigma



    def mu_process_tf(self, t, x):
        # first d dimension of x corresponds to stock price, second d to the variance process
        M = tf.shape(x)[0]
        d = tf.shape(x)[1]

        log_stock_price = x[:, 0: self.m]
        nu = x[:, self.m:]

        log_stock_drift = self.mu - 0.5 * x[:, self.m:]  # slicing is closed, open
        vol_drift = self.kappa * (self.nu_bar - nu) * tf.ones((M, self.m), dtype=tf.keras.backend.floatx())

        return tf.concat([log_stock_drift, vol_drift], axis=-1)

    def sigma_process_tf(self, t, x):
        # sigma should be the covariance matrix:
        M = tf.shape(x)[0]
        d = tf.shape(x)[1]

        stock_price = x[:, 0: self.m]
        nu = x[:, self.m:]

        stock_diagonal = tf.pow(nu, 0.5)
        vol_diagonal = self.xi * tf.pow(nu, 0.5)

        diagonal = tf.concat([stock_diagonal, vol_diagonal], axis=-1)  # M x (2d)

        correlated_sigma = tf.linalg.diag(diagonal)
        uncorrelated_sigma = tf.einsum("Mij, jk -> Mik", correlated_sigma, self.cholesky)

        return uncorrelated_sigma

    def inverse_sigma_process_tf(self, t, x):
        # sigma should be the covariance matrix:
        stock_price = x[:, 0: self.m]
        nu = self.f_3_truncation(x[:, self.m:])

        stock_diagonal = tf.pow(nu, 0.5)
        vol_diagonal = self.xi * tf.pow(nu, 0.5)

        diagonal = tf.concat([stock_diagonal, vol_diagonal], axis=-1)  # M x (2d)
        inverse_diagonal_part = tf.linalg.diag(tf.pow(diagonal, -1))

        return tf.einsum("jk, Mki -> Mji", self.inv_cholesky, inverse_diagonal_part)
class BermudanGeometricHeston(HestonBaseClass):
    def __init__(self, eqn_config):
        super(BermudanGeometricHeston, self).__init__(eqn_config)
        self.is_smoothed = self.eqn_config.parameter_config.is_smoothed

        self.call_or_put = self.eqn_config.parameter_config.call_or_put
        if self.call_or_put.lower() == 'call':
            self.indicator_call_or_put = 1.0
        elif self.call_or_put.lower() == 'put':
            self.indicator_call_or_put = -1.0
        else:
            raise ValueError()

        self.J = 1
        self.sigma = self.eqn_config.parameter_config.sigma
        self.q = self.eqn_config.parameter_config.q
        self.r = self.eqn_config.parameter_config.r
        self.x_0 = self.eqn_config.parameter_config.X_0
        self.K = self.eqn_config.parameter_config.K
        DTYPE = tf.keras.backend.floatx()
        self.nu_0 = tf.cast(self.eqn_config.parameter_config.nu_0, DTYPE).numpy()
        self.mu = self.r - self.q

        # correlation_matrix = np.zeros([self.d, self.d], dtype=self.DTYPE)
        # for i in range(self.d):
        #     for j in range(i, self.d):
        #         if i < self.m:
        #             # i is a stock
        #             if j < self.m:
        #                 if i == j:
        #                     correlation_matrix[i, j] = 0.5
        #                 else:
        #                     correlation_matrix[i, j] = 0.25
        #             else:
        #                 if np.abs(i - j) == self.m:
        #                     # own volatility
        #                     correlation_matrix[i, j] = -0.64
        #                 else:
        #                     # other vols
        #                     correlation_matrix[i, j] = 0.0
        #         else:
        #             if i == j:
        #                 correlation_matrix[i, j] = 0.5
        #             elif j > self.m:
        #                 correlation_matrix[i, j] = 0.05
        # correlation_matrix += correlation_matrix.T
        # self._get_cholesky(correlation_matrix)



    def g_tf(self, t, x):
        stock_price = x[..., 0: self.m]  # first d entry correspond to stock, second d to volatilities
        payoff = self.K - tf.reduce_prod((stock_price) ** (1 / self.m), axis=-1, keepdims=True)
        return tf.where(payoff > 0, payoff, 0)

    def f_tf(self, t, x, y, z):
        return -self.r * y  # let's first do this under the risk neutral measure

class LargeBSPortfolio(SystemReflectedFBSDE):
    """
    """
    def __init__(self, eqn_config):


        super(LargeBSPortfolio, self).__init__(eqn_config)

        self.J = 1

        self.rho = self.eqn_config.parameter_config.rho
        self.mu = tf.cast(tf.constant(self.eqn_config.parameter_config.mu), self.DTYPE)
        self.sigma = tf.cast(tf.constant(self.eqn_config.parameter_config.sigma), self.DTYPE)
        self.q = self.eqn_config.parameter_config.q
        self.r = self.eqn_config.parameter_config.r
        self.x_0 = self.eqn_config.parameter_config.X_0
        self.K = self.eqn_config.parameter_config.K

        correl_mtx = eqn_config.parameter_config.rho * tf.ones(shape=[self.d, self.d],
                                                               dtype=tf.keras.backend.floatx())
        self.correlation_mtx = tf.linalg.set_diag(correl_mtx, tf.ones(self.d, dtype=tf.keras.backend.floatx()))
        self._get_cholesky(self.correlation_mtx)



        self.is_smoothed = self.eqn_config.parameter_config.is_smoothed

        self.call_or_put = self.eqn_config.parameter_config.call_or_put
        if self.call_or_put.lower() == 'call':
            self.indicator_call_or_put = 1.0
        elif self.call_or_put.lower() == 'put':
            self.indicator_call_or_put = -1.0
        else:
            raise ValueError()

        self.nabla_sigma_const = np.zeros([self.d, self.d, self.d]).astype(tf.keras.backend.floatx())
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    if i == k:
                        self.nabla_sigma_const[i, j, k] = self.sigma[k] * self.cholesky[k, j]

        self.nabla_sigma_const = tf.constant(self.nabla_sigma_const)

    def get_hessian_from_gamma(self, t, x, grad_u, gamma, is_held_at_t):
        # # # is_held_at_t is a tensor of shape M x J valued 1.0 if the option has not been exercised at any s<t, and 0.0 otherwise
        # # # gamma = (hess-y^T x sigma)^T + grad-y x grad-sigma


        nabla_u_nabla_sigma = tf.einsum("Md, dqm -> Mqm",
                                        grad_u, self.nabla_sigma_const)

        hess = tf.einsum("Mij, Mik -> Mjk",
                         self.inverse_sigma_process_tf(t, x),
                         tf.einsum("MJ, MJik -> Mik", is_held_at_t, gamma) - nabla_u_nabla_sigma)

        return hess

    def grad_u_from_reduced(self, t, x, grad_u_bar):
        # u_bar should be the same as u, but depending on m instead of x where m is the geometric average of the
        # coordinates of x

        m = tf.reduce_prod(tf.pow(x, 1 / self.d), axis=-1, keepdims=True)
        grad_m = m / self.d * tf.pow(x, -1)

        return grad_u_bar * grad_m

    def hess_u_from_reduced(self, t, x, grad_u_bar, hess_u_bar):
        m = tf.reduce_prod(tf.pow(x, 1 / self.d), axis=-1, keepdims=True)
        grad_m = m / self.d * tf.pow(x, -1)

        mtx1 = tf.einsum("Mi, Mdq -> Mdq", -m / self.d, tf.linalg.diag(tf.pow(x, -2)))
        mtx2 = tf.einsum('Mi, Mdq -> Mdq',
                         m / (self.d ** 2),
                         tf.pow(tf.einsum("Mi, Mj -> Mij", x, x), -1))
        hess_m = mtx1 + mtx2


        term1 = tf.einsum("Mi, Mdq -> Mdq", grad_u_bar, hess_m)
        term2 = tf.einsum("Mi, Mdq -> Mdq", hess_u_bar, tf.einsum("Mi, Mj -> Mij", grad_m, grad_m))
        hess_u = term1 + term2

        return hess_u


    def mu_process_tf(self, t, x):
        return tf.einsum("i, Mi -> Mi", self.mu, tf.pow(x, 1))

    def f_tf(self, t, x, y, z):
        return -self.r * tf.pow(y, 1) - tf.einsum("i, MJi -> MJ",
                                                  (self.mu - (self.r - self.q)) / self.sigma,
                                                  tf.einsum("MJd, dq -> MJq", z, self.inv_cholesky))
    def sigma_process_tf(self, t, x):
        uncorrelated_sigma = tf.einsum("Mij, jk -> Mik",
                                       tf.linalg.diag(tf.einsum("i, Mi -> Mi", self.sigma, x)), self.cholesky)

        return uncorrelated_sigma

    def inverse_sigma_process_tf(self, t, x):
        return tf.einsum("ij, Mjk-> Mik",
                         self.inv_cholesky,
                         tf.linalg.diag(tf.einsum("i, Mi -> Mi", self.sigma, x) ** (-1)))


    def sample_step_malliavin(self, dw, x, n):
        return self.sigma_process_tf(self.t[n + 1], x[..., n + 1])
    @tf.function
    def sample(self, M):
        """ dX_t = mu * dt + sigma * dW_t, X_0=x"""
        DTYPE = tf.keras.backend.floatx()
        dw_sample = tf.random.normal(shape=[M, self.m, self.N], dtype=DTYPE) * np.sqrt(self.delta_t)
        # drift_sample = tf.ones(shape=[M, self.m, self.N], dtype=DTYPE) * (self.mu - self.sigma ** 2 / 2) * self.delta_t
        drift_sample = tf.einsum("m, MmN -> MmN",
                                 (self.mu - self.sigma ** 2 / 2) * self.delta_t,
                                 tf.ones(shape=[M, self.m, self.N], dtype=DTYPE))
        initial_part = tf.math.log(tf.cast(self.x_0, dtype=tf.keras.backend.floatx())) * tf.ones(shape=[M, self.m, 1],
                                                                                                 dtype=DTYPE)
        SIGMA = tf.einsum("ij, jk -> ik", tf.linalg.diag(self.sigma), self.cholesky)
        simulation_part = drift_sample + tf.einsum("ij, Mjn -> Min", SIGMA, dw_sample)

        x_sample = tf.concat([initial_part, simulation_part], axis=-1)
        x_sample = tf.cumsum(x_sample, axis=-1)
        x_sample = tf.exp(x_sample)

        return dw_sample, x_sample
    def g_tf(self, t, x):
        geom_mean_x = tf.reduce_prod(tf.pow(x, 1 / self.d), axis=-1, keepdims=True)
        payoff = self.indicator_call_or_put * (geom_mean_x - self.K)
        return tf.math.maximum(payoff, 0)

class CustomBlackScholesPortfolio(LargeBSPortfolio):
    """
    """
    def __init__(self, eqn_config):
        super(CustomBlackScholesPortfolio, self).__init__(eqn_config)
        self.d = eqn_config.d
        correl_mtx = eqn_config.parameter_config.rho * tf.ones(shape=[self.d, self.d],
                                                               dtype=tf.keras.backend.floatx())
        self.correlation_mtx = tf.linalg.set_diag(correl_mtx, tf.ones(self.d, dtype=tf.keras.backend.floatx()))
        self._get_cholesky(self.correlation_mtx)

        self.J = self.d + 5
        self.K_1 = self.K
        self.K_2 = self.K * 1.2
        self.K_3 = self.K * 0.8
        self.K_4 = self.K * 1.5

        self.K_5 = self.K * 0.5

        self.cash_or_nothing_lower = 0.5 * self.K
        self.cash_or_nothing_upper = 1.5 * self.K

        if self.m % 2 != 0:
            raise ValueError("Even number of assets needed")


    def g_tf(self, t, x):
        payoff_1 = self.K_1 - tf.reduce_prod(tf.pow(x[..., 0: self.m], 1 / self.m), axis=-1, keepdims=True)
        g_1 = tf.math.maximum(payoff_1, 0)

        payoff_2 = self.K_2 - tf.reduce_mean(x[..., 0: int(self.m / 2)], axis=-1, keepdims=True)
        g_2 = tf.math.maximum(payoff_2, 0)

        payoff_3 = tf.reduce_max(x[..., int(self.m / 2): ], axis=-1, keepdims=True) - self.K_3
        g_3 = tf.math.maximum(payoff_3, 0)

        payoff_4 = tf.where(x[..., 0: self.m] >= self.cash_or_nothing_lower,
                          tf.where(x[..., 0: self.m] <= self.cash_or_nothing_upper, 1.0, 0.0),
                          0.0)
        g_4 = tf.reduce_prod(payoff_4, axis=-1, keepdims=True)

        payoff_5 = self.K_5 - tf.reduce_min(x[..., int(self.m / 2): ], axis=-1, keepdims=True)
        g_5 = tf.math.maximum(payoff_5, 0)

        G = tf.concat([g_1, g_2, g_3, g_4, g_5], axis=-1)

        # g_i = tf.math.maximum(x - self.K_4, 0)
        G = tf.concat([G, tf.math.maximum(x - self.K_4, 0)], axis=-1)

        # for i in range(self.m):
        #     g_i = tf.math.maximum(x[..., i: i+1] - self.K_4, 0)
        #     G = tf.concat([G, g_i], axis=-1)

        return G


class CustomBlackScholesPortfolioATM(CustomBlackScholesPortfolio):
    def __init__(self, eqn_config):
        super(CustomBlackScholesPortfolioATM, self).__init__(eqn_config)

        self.J = self.d + 5
        self.K_1, self.K_2, self.K_3, self.K_4, self.K_5 = self.K, self.K, self.K, self.K, self.K

        self.cash_or_nothing_lower = 0.5 * self.K
        self.cash_or_nothing_upper = 1.5 * self.K
