import scipy.linalg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import time
import os
from lib.SubNet import SubNet
from lib.misc.learning_rate_schedules import ChenSchedule

def custom_adam_optimizer(train_size, batch_size, epochs, initial_learning_rate, decay_rate, lower_bound_learning_rate):
    '''
    default custom optimizer used in all classes below;
    in case the user would want to change optimizer this declaration function is the only piece of code to be rewritten
    :param train_size: size of the Monte Carlo sample used for training
    :param batch_size: (constant) batch size used in training
    :param epochs: num of times each sample path is processed in the optimizatino
    :param initial_learning_rate: initial_learning_rate parameter of ChenSchedule
    :param decay_rate: decay_rate parameter of ChenSchedule
    :param lower_bound_learning_rate: lower_bound_learning_rate parameter of ChenSchedule
    :return: declared Adam optimizer
    '''
    bn = train_size / batch_size
    num_of_batches = np.int(np.modf(bn)[1]) if np.modf(bn)[0] == 0 else np.int(np.modf(bn)[1] + 1)
    total_num_of_iterations = epochs * num_of_batches

    optimizer = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(initial_learning_rate, decay_rate,
                                                                    lower_bound_learning_rate, total_num_of_iterations),
                                         epsilon=1e-8)

    return optimizer

class BackwardBSDESolverBaseClass(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        # self.t = np.arange(self.bsde.num_time_interval + 1) * self.bsde.delta_t
        self.t = self.bsde.t
        self.t = self.t.astype(tf.keras.backend.floatx())
        self.Y = [None] * len(self.t)  # these are going to be a list of SubNets which can be called over the whole
        self.Z = [None] * len(self.t)
        self.G = [None] * len(self.t)  # models may have gamma estimates for t = 0

        self.is_has_gamma = False  # Malliavin models may have gammas as well
        self.is_gamma_autodiff = True

        self.is_collect_future = False
        self.train_collected_future = None  # will be needed for the multistep approach,
        # to avoid recalculating already processed time steps in the minibatch training
        self.valid_collected_future = None  # None will default to not using them at all
        # time points at the start of each iteration step
        # this is a dict with keys: {'Y': ndarray, 'Z': ndarray, 'grad_Y': ndarray, 'jac_Z': ndarray}

        if not hasattr(self.net_config, "layer_width"):
            self.net_config.layer_width = 100

        if self.net_config.num_hiddens == 'default':
            self.net_config.num_hiddens = [self.bsde.dim + self.net_config.layer_width,
                                           self.bsde.dim + self.net_config.layer_width]
            logging.info("@BackwardBSDESolverBaseClass:init: default network architecture has depth 2 with 100+dim many neurons in each layer")
        elif "default_" in self.net_config.num_hiddens:
            num = self.net_config.num_hiddens.split("_")[1]
            try:
                num = int(num)
            except:
                raise ValueError("Integer not understood.")
            if num < 0:
                raise ValueError("Num hidden layers>0!")
            self.net_config.num_hiddens = [self.bsde.m + self.net_config.layer_width] * num
            logging.info(
                "@BackwardBSDESolverBaseClass:init: default network architecture has depth %d with 100+dim many neurons in each layer"%num)
        logging.info(self.net_config.num_hiddens)


        if self.net_config.train_size is not None:
            logging.info("@BackwardBSDESolverBaseClass:init: assuming fetch training")
            bn = self.net_config.train_size / self.net_config.batch_size
            no_of_batches = np.int(np.modf(bn)[1]) if np.modf(bn)[0] == 0 else np.int(np.modf(bn)[1] + 1)
            self.no_of_batches = no_of_batches
        else:
            self.no_of_batches = 1  # new batch fetched in each iteration


        if not hasattr(self.net_config, "is_fk_y"):
            self.is_fk_y = False
            logging.info("@BackwardBSDESolverBaseClass:init:Feynman-Kac: Y: no FK rule provided (default to false)")
        else:
            self.is_fk_y = self.net_config.is_fk_y
            logging.info("@BackwardBSDESolverBaseClass:init:Feynman-Kac: Y: using Y ~ sigma^{-1} x Z")
        if not hasattr(self.net_config, "is_fk_z"):
            self.is_fk_z = False
            logging.info("@BackwardBSDESolverBaseClass:init:Feynman-Kac: Z: no FK rule provided (default to false)")
        else:
            self.is_fk_z = self.net_config.is_fk_z
            logging.info("@BackwardBSDESolverBaseClass:init:Feynman-Kac: Z: using Z ~ sigma x nabla_x Y")
        if self.is_fk_y and self.is_fk_z:
            raise ValueError("@BackwardBSDESolverBaseClass:init:Feynman-Kac: conflicting FK rule, either Y or Z")

        self.is_z_autodiff = False

        if not hasattr(self.net_config, "pretrained_num_iterations"):
            self.net_config.pretrained_num_iterations = self.net_config.num_iterations
            logging.info("@BackwardBSDESolverBaseClass:init: no early termination for inner time points")

        if not hasattr(self.net_config, "is_t0_explicit_init"):
            self.net_config.is_t0_explicit_init = True
            logging.info("@BackwardBSDESolverBaseClass:init: doing t0 initialization according to explicit schemes by default")

        if not hasattr(self.eqn_config.discretization_config, "theta_y"):
            self.theta_y = None
        else:
            self.theta_y = self.eqn_config.discretization_config.theta_y

        if not hasattr(self.eqn_config.discretization_config, "theta_z"):
            self.theta_z = None
        else:
            self.theta_z = self.eqn_config.discretization_config.theta_z

        if not hasattr(self.net_config, "is_nn_at_t0"):
            self.is_nn_at_t0 = False
            logging.info("@BackwardBSDESolverBaseClass:init: using single parameter estimations at t=0")
        else:
            self.is_nn_at_t0 = self.net_config.is_nn_at_t0

        self.is_trained = False

    # # # Yn
    def loss_y_n(self, n, inputs, training):
        raise NotImplementedError

    def grad_y_n(self, n, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_y_n(n, inputs, training)
        grad = tape.gradient(loss, self.model_y_n.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_y_n(self, n, inputs):
        grad = self.grad_y_n(n, inputs, training=True)
        self.optimizer_y_n.apply_gradients(zip(grad, self.model_y_n.trainable_variables))

    # # # Y0
    def loss_y_0(self, n, inputs, training):
        raise NotImplementedError

    def grad_y_0(self, n, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_y_0(n, inputs, training)
        grad = tape.gradient(loss, self.model_y_0.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_y_0(self, n, inputs):
        grad = self.grad_y_0(n, inputs, training=True)
        self.optimizer_y_0.apply_gradients(zip(grad, self.model_y_0.trainable_variables))

    # # # Zn
    def loss_z_n(self, n, inputs, training):
        raise NotImplementedError

    def grad_z_n(self, n, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_z_n(n, inputs, training)
        grad = tape.gradient(loss, self.model_z_n.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_z_n(self, n, inputs):
        grad = self.grad_z_n(n, inputs, training=True)
        self.optimizer_z_n.apply_gradients(zip(grad, self.model_z_n.trainable_variables))

    # # # Z0
    def loss_z_0(self, n, inputs, training):
        raise NotImplementedError

    def grad_z_0(self, n, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_z_0(n, inputs, training)
        grad = tape.gradient(loss, self.model_z_0.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step_z_0(self, n, inputs):
        grad = self.grad_z_0(n, inputs, training=True)
        self.optimizer_z_0.apply_gradients(zip(grad, self.model_z_0.trainable_variables))

    def train_minibatch(self, output_dir):
        raise NotImplementedError

    def train(self, output_dir):
        raise NotImplementedError

    def get_targets_y(self, n, data, collected_future):
        raise NotImplementedError

    def get_inputs_y(self, n, data, collected_future):
        raise NotImplementedError

    def get_targets_z(self, n, data, collected_future):
        raise NotImplementedError

    def get_inputs_z(self, n, data, collected_future):
        raise NotImplementedError

    def model_snapshot_y_n(self, n, dummy_input):
        raise NotImplementedError

    def model_snapshot_y_0(self, n, dummy_input):
        raise NotImplementedError

    def model_snapshot_z_n(self, n, dummy_input):
        raise NotImplementedError

    def model_snapshot_z_0(self, n, dummy_input):
        raise NotImplementedError

    @staticmethod
    def _reset_optimizer(optimizer):
        '''
        overrides all parameters of an optimizer with 0s (such as: processed num of batches, learning rate, etc.)
        :param optimizer: tested mostly with ADAM -> raise ValueError for other optimizers
        :return: 0 (optimizer's state is modified on the fly)
        '''
        if optimizer.name != "Adam":
            raise ValueError
        for param in optimizer.variables:
            param.assign(np.zeros(param.numpy().shape))
        logging.info("@optimizer state reset to Zero.")
        return 0

    def data_iterator(self, tensor_slices):
        '''
        gets a tuple of training inputs (input1, input2, ..., target), creates a tf dataset with batches, shuffles,
         prefetching, etc.
        :param tensor_slices: tuple of training inputs (input1, input2, ..., target)
        :return: dataset object
        '''
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slices)
        dataset = dataset.shuffle(self.net_config.train_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.net_config.batch_size)  # rewritten to batch_size_eff
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def initialize_collected_future(self, data):
        dw, x = data
        M = dw.shape[0]
        y = np.zeros(shape=[M, 1, self.bsde.N + 1], dtype=tf.keras.backend.floatx())
        z = np.zeros(shape=x.shape, dtype=tf.keras.backend.floatx())
        grad_y = np.zeros(shape=x.shape, dtype=tf.keras.backend.floatx())
        grad_z = np.zeros(shape=[M, self.bsde.dim, self.bsde.dim, self.bsde.N + 1],
                          dtype=tf.keras.backend.floatx())

        retval_dict = {'Y': y, 'Z': z, 'grad_Y': grad_y, 'jac_Z': grad_z}

        return retval_dict

    def update_collected_future(self, n, data, collected_future):
        # # # update collected future with this time step: to be called after optimization
        dw, x = data

        collected_future['Y'][:, :, n] = self.Y[n](x[:, :, n], False).numpy()
        raise NotImplementedError("not yet rewritten")

        collected_future['grad_Y'][:, :, n], collected_future['jac_Z'][:, :, :, n] = self.get_grads_through_batching(n,
                                                                                                                     x)
        if (self.is_z_autodiff) and (n < self.bsde.N - 1):
            sigma_n = self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.t[n], x[:, :, n]).numpy()
            collected_future['Z'][:, :, n] = np.einsum('aij, aj -> ai', sigma_n, collected_future['grad_Y'][:, :, n])
        else:
            collected_future['Z'][:, :, n] = self.Z[n](x[:, :, n], False).numpy()

        return 0

    def validation_batching(self, func, n, inputs, training, chunk_size=None):
        """
        :param func: function pointer pointing to the loss function to evaluate through batching
        :param n:
        :param inputs:
        :param training:
        :return:
        """
        if chunk_size is None:
            chunk_size = self.net_config.batch_size

        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(chunk_size)  # no need to shuffle or anything
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        valid_loss = 0
        for valid_batch in dataset:
            valid_loss += func(n, valid_batch, training)

        return valid_loss

    def get_grads_through_batching(self, n, x, chunk_size=1024):
        # # # Collects gradY[n](x_n), gradZ[n](x_n) from the future, in chunk size bits to avoid memory overflow
        # # # the two booleans determine whether gradY[n] should be approximated by sigma^-1 * Z
        # # # and whether gradZ should be approximated by (grad sigma) * gradY + sigma * hessY
        if self.is_fk_y:
            logging.info("@! gradY = sigma^(-1) * Z Feynman-Kac approximation used")
            if self.is_fk_z:
                raise ValueError('### Suspectedly inconsistent FK booleans. Prefer either Y or Z.')

        if self.is_fk_z:
            logging.info("@! gradZ = (grad sigma) * gradY + sigma * hessY Feynman-Kac approximation used")

        if self.bsde.dim > 1:
            logging.info("@Gradient chunk size automatically adjusted to 1024 / d^2")
            chunk_size = np.ceil(8192 / self.bsde.dim ** 2)
            chunk_size = np.int(2 ** np.ceil(np.log(chunk_size) / np.log(2)))

        if self.bsde.dim > 50:
            logging.info("@Gradient chunk size automatically adjusted to M / (d^2 / 50^2)")
            chunk_size = np.ceil(self.net_config.train_size / (self.bsde.dim ** 2 / 50 ** 2))
            chunk_size = np.int(2 ** np.ceil(np.log(chunk_size) / np.log(2)))
            chunk_size = self.net_config.batch_size

        x_n = x[:, :, n]
        M = x_n.shape[0]
        cn = M / chunk_size
        number_of_chunks = np.int(np.modf(cn)[1]) if np.modf(cn)[0] == 0 else np.int(np.modf(cn)[1] + 1)

        # initialize
        grad_y_n = np.zeros(x_n.shape, dtype=tf.keras.backend.floatx())
        jac_z_n = np.zeros(shape=[M, self.bsde.dim, self.bsde.dim], dtype=tf.keras.backend.floatx())
        for i in range(number_of_chunks):
            idx_low = i * chunk_size
            idx_up = (i + 1) * chunk_size
            if i == number_of_chunks - 1:
                x_n_chunk = x_n[idx_low:, :]
            else:
                x_n_chunk = x_n[idx_low:idx_up, :]
            if n == self.bsde.N:
                grad_y_n_chunk = self.bsde.gradx_g_tf(self.bsde.T, x_n_chunk)
                x_n_chunk_tf = tf.constant(x_n_chunk)
                jac_z_n_chunk = self.get_last_jac_z_through_autodiff(x_n_chunk_tf)
                del x_n_chunk_tf
            else:
                if self.is_fk_y:
                    inv_sigma_chunk = self.bsde.ForwardDiffusion.inverse_sigma_process_tf(self.bsde.t[n], x_n_chunk)
                    grad_y_n_chunk = tf.einsum('aij, aj -> ai', inv_sigma_chunk, self.Z[n](x_n_chunk, False))  # FK
                else:
                    grad_y_n_chunk = self.Y[n].grad_call(x_n_chunk, False)

                if self.is_fk_z:
                    # # # nabla Zt = nabla (sigma nabla Yt) = (nabla sigma) (nabla Yt) + sigma nabla^T nabla Yt
                    term1 = tf.einsum('aijk, ak -> aij',
                                      self.bsde.ForwardDiffusion.nabla_sigma_process_tf(self.bsde.t[n], x_n_chunk),
                                      grad_y_n_chunk)
                    term2 = tf.einsum('aij, ajk -> aik',
                                      self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.t[n], x_n_chunk),
                                      self.Y[n].hessian_call(x_n_chunk, False))
                    jac_z_n_chunk = term1 + term2
                else:
                    jac_z_n_chunk = self.Z[n].jacobian_call(x_n_chunk, False)

            if i == number_of_chunks - 1:
                grad_y_n[idx_low: , :] = grad_y_n_chunk.numpy()
                jac_z_n[idx_low: , :, :] = jac_z_n_chunk.numpy()
            else:
                grad_y_n[idx_low:idx_up, :] = grad_y_n_chunk.numpy()
                jac_z_n[idx_low:idx_up, :, :] = jac_z_n_chunk.numpy()
        return grad_y_n, jac_z_n

    @tf.function(experimental_relax_shapes=True)
    def get_last_jac_z_through_autodiff(self, x_n_chunk_tf):
        # print("get_last_z_through_autodiff: retracing")
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x_n_chunk_tf)
            z_tmp = self.Z[-1](x_n_chunk_tf, False)
        jac_z_n_chunk = tape.batch_jacobian(z_tmp, x_n_chunk_tf, experimental_use_pfor=True)
        return jac_z_n_chunk

    def plot_Yn(self, n, sample, path, markersize=3):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.bsde.dim > 1:
            logging.info("@!!: Multi-D BSDE encountered: no plotting")
            return 1
        dw_sample, x_sample = sample
        if n == 0:
            if not self.is_nn_at_t0:
                logging.info("@plot_Yn: n==0 not plottable for single variable parametrization")
                return 0
            else:
                x = x_sample[:, :, 1]
                logging.info("@plot_Yn: neural net at t=0: plotting over a sample of X_1^\pi")
        else:
            x = x_sample[:, :, n]

        if self.Y[n] is not None:
            if n < self.bsde.N:
                y = self.Y[n](x, False)
            y = self.Y[n](x, False)  # the user is expected to format the payoff accordingly with lambda functions
            fig, ax = plt.subplots()
            ax.scatter(x, y, label='NN', s=markersize)
            title = 'T=%.2e, N=%u' % (self.bsde.T, self.bsde.N)
            if self.bsde.is_y_theoretical:
                y_anal = self.bsde.y_analytical(self.t[n], x)
                ax.scatter(x, y_anal, label='analytical', s=markersize)
                l2err = np.linalg.norm(y_anal - y) / np.linalg.norm(y_anal)
                title += ', Rel-L2-Error: %.3e' % l2err
            plt.xlabel('$X_t$')
            plt.ylabel('$Y_t(X_{t})$')
            plt.suptitle('$Y(x, t=t_n=%.2e), n=%u$' % (self.t[n], n))
            plt.title(title)
            plt.grid()
            plt.legend()
            fig.savefig(path + '/Y_' + str(n) + '.png', format='png')
            plt.close(fig)
        return 0

    def plot_Zn(self, n, sample, path, markersize=3):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.bsde.dim > 1:
            logging.info("@!!: Multi-D BSDE encountered: no plotting")
            return 1
        dw_sample, x_sample = sample
        if n == 0:
            if not self.is_nn_at_t0:
                logging.info("@plot_Zn: n==0 not plottable for single variable parametrization")
                return 0
            else:
                x = x_sample[:, :, 1]
                logging.info("@plot_Yn: neural net at t=0: plotting over a sample of X_1^\pi")
        else:
            x = x_sample[:, :, n]

        if self.Z[n] is not None:
            z = self.Z[n](x, False)  # the user is expected to format the payoff accordingly with lambda functions
            if self.is_z_autodiff:
                z = tf.einsum('aij, aj -> ai',
                              self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.t[n], x),
                              z)
            fig, ax = plt.subplots()
            ax.scatter(x, z, label='NN', s=markersize)
            title = 'T=%.2e, N=%u' % (self.bsde.T, self.bsde.N)
            if self.bsde.is_y_theoretical:
                z_anal = self.bsde.z_analytical(self.t[n], x)
                logging.info("@plot_zn: z should be z, not grad y")
                # if not self.bsde.is_sigma_in_control:
                #     # # # Z is sigma(t, X) nabla Y(t, x)
                #     z_anal = tf.einsum('ink,ik->in',
                #                        self.bsde.ForwardDiffusion.sigma_process_tf(self.t[n], x),
                #                        tf.constant(z_anal)).numpy()  # matrix-vector multiplication over batches
                ax.scatter(x, z_anal, label='analytical', s=markersize)
                l2err = np.linalg.norm(z_anal - z) / np.linalg.norm(z_anal)
                title += ', Rel-L2-Error: %.3e' % l2err
            plt.xlabel('$X_t$')
            plt.ylabel('$Z_t(X_{t})$')
            plt.suptitle('$Z(x, t=t_n=%.2e), n=%u$' % (self.t[n], n))
            plt.title(title)
            plt.grid()
            plt.legend()
            fig.savefig(path + '/Z_' + str(n) + '.png', format='png')
            plt.close(fig)
        return 0

    def plot_Gn(self, n, sample, path, markersize=3):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.bsde.dim > 1:
            logging.info("@!!: Multi-D BSDE encountered: no plotting")
            return 1
        dw_sample, x_sample = sample
        if n == 0:
            if not self.is_nn_at_t0:
                logging.info("@plot_Zn: n==0 not plottable for single variable parametrization")
                return 0
            else:
                x = x_sample[:, :, 1]
                logging.info("@plot_Yn: neural net at t=0: plotting over a sample of X_1^\pi")
        else:
            x = x_sample[:, :, n]

        if self.Z[n] is not None:
            if n == self.bsde.N:
                gamma = self.get_last_jac_z_through_autodiff(x)
            else:
                if not self.is_z_autodiff:
                    if self.is_gamma_autodiff:
                        gamma = self.Z[n].jacobian_call(x, False)  # Mxdxd
                    else:
                        gamma = self.G[n](x, False)
                else:
                    x_tf = tf.constant(x)
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        sigma_tf = self.bsde.ForwardDiffusion.sigma_process_tf(self.t[n], x_tf)
                        z_tf = tf.einsum("Mij, Mj -> Mi", sigma_tf, self.Z[n](x_tf, False))
                    gamma = tape.batch_jacobian(z_tf, x_tf).numpy()
            fig, ax = plt.subplots()
            ax.scatter(x, gamma[:, :, 0], label='NN', s=markersize)
            title = 'T=%.2e, N=%u' % (self.bsde.T, self.bsde.N)
            if self.bsde.is_y_theoretical:
                gamma_anal = self.bsde.gamma_analytical(self.t[n], x)
                ax.scatter(x, gamma_anal, label='analytical', s=markersize)
                l2err = np.linalg.norm(gamma_anal - gamma) / np.linalg.norm(gamma_anal)
                title += ', Rel-L2-Error: %.3e' % l2err
            plt.xlabel('$X_t$')
            plt.ylabel('$\Gamma_t(X_{t})$')
            plt.suptitle('$\Gamma(x, t=t_n=%.2e), n=%u$' % (self.t[n], n))
            plt.title(title)
            plt.grid()
            plt.legend()
            fig.savefig(path + '/Gamma_' + str(n) + '.png', format='png')
            plt.close(fig)
        return 0

    def save_step_history(self, n, hist, path):
        '''
        saves training histories of a completed time step
        :param n: time step
        :param hist: dictionary (possibly) containing keys: {'Y': [], 'Z': [], 'G': []} corresponding to the trainings
        :param path: output path to drop csvs to
        :return: 0
        '''

        if not os.path.exists(path):
            os.makedirs(path)
        if hist[n] is not None:
            for process in hist[n].keys():
                fn = path + '/' + process + '_train_history_' + str(n) + '.csv'
                to_save = hist[n][process]
                fmt = ['%d', '%.6e', '%.10e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e']

                np.savetxt(fn, to_save,
                           fmt=fmt,
                           delimiter=",",
                           header='step, learning_rate, loss_function, sampling_time, target_gathering_time, valid_loss_time, gradient_step_time, iteration_time, step_time, cumulative_elapsed_time',
                           comments='')
        else:
            warnings.warn('History[%u] not found' % n)
        return 0

    def save_models_n(self, n, path, name_convention='_weights'):
        if not os.path.exists(path):
            os.makedirs(path)
        if n == 0:
            if not self.is_nn_at_t0:
                # these are not networks, but single parameters
                np.savetxt(path + "/Y_0.csv", self.Y[0].numpy())
                np.savetxt(path + "/Z_0.csv", self.Z[0].numpy())
                if self.is_has_gamma:
                    np.savetxt(path + '/G_0.csv', self.G[0].numpy()[0, :, :])  # first axis is dummy
            else:
                if self.Y[n] is None:
                    warnings.warn('Y[%u] has not been trained, therefore will not be saved' % n)
                else:
                    self.Y[n].save_weights(path + '/Y_' + str(n) + name_convention, save_format='tf')
                if self.Z[n] is None:
                    warnings.warn('Z[%u] has not been trained, therefore will not be saved' % n)
                else:
                    if not self.is_z_autodiff:
                        self.Z[n].save_weights(path + '/Z_' + str(n) + name_convention, save_format='tf')
                    else:
                        warnings.warn(
                            'Z[%u] has been taken as automatic differentiation, therefore will not be saved' % n)
                if self.G[n] is None:
                    warnings.warn('Gamma[%u] has not been trained, therefore will not be saved' % n)
                else:
                    if not self.is_gamma_autodiff:
                        self.G[n].save_weights(path + '/G_' + str(n) + name_convention, save_format='tf')
        elif n == self.bsde.N:
            pass  # terminal condition => no models to pass
        else:
            if self.Y[n] is None:
                warnings.warn('Y[%u] has not been trained, therefore will not be saved' % n)
            else:
                self.Y[n].save_weights(path + '/Y_' + str(n) + name_convention, save_format='tf')
            if self.Z[n] is None:
                warnings.warn('Z[%u] has not been trained, therefore will not be saved' %n)
            else:
                if not self.is_z_autodiff:
                    self.Z[n].save_weights(path + '/Z_' + str(n) + name_convention, save_format='tf')
                else:
                    warnings.warn('Z[%u] has been taken as automatic differentiation, therefore will not be saved' % n)
            if self.G[n] is None:
                warnings.warn('Gamma[%u] has not been trained, therefore will not be saved' % n)
            else:
                if not self.is_gamma_autodiff:
                    self.G[n].save_weights(path + '/G_' + str(n) + name_convention, save_format='tf')
        return 0

    def load_from_file(self, path, name_convention='_weights'):
        print(path)
        dummy_input = np.random.normal(size=[5, self.bsde.d])  # this will only be used to reinitialize the signatures
        if self.is_trained:
            warnings.warn('This model has just been trained, are you sure you want to override its parameters?')
            return -1

        # return "I am in the right place"

        # # # naming convention should be the same as with save
        # # n = 0
        if not self.is_nn_at_t0:
            raise NotImplementedError
            filename = 'Y_0.csv'
            if filename not in os.listdir(path):
                warnings.warn('Y_0 not found in folder', RuntimeWarning)
            else:
                self.Y[0] = tf.Variable(np.reshape(np.loadtxt(path + '/' + filename), newshape=[1]),
                                        dtype=tf.keras.backend.floatx())
            filename = 'Z_0.csv'
            if filename not in os.listdir(path):
                warnings.warn('Z_0 not found in folder', RuntimeWarning)
            else:
                self.Z[0] = tf.Variable(np.reshape(np.loadtxt(path + '/' + filename), newshape=[J, self.bsde.d]),
                                        dtype=tf.keras.backend.floatx())

            if self.is_has_gamma:
                filename = 'G_0.csv'
                if filename not in os.listdir(path):
                    warnings.warn('G_0 not found in folder', RuntimeWarning)
                else:
                    self.G[0] = tf.Variable(np.reshape(np.loadtxt(path + '/' + filename),
                                                       newshape=[1, self.bsde.dim, self.bsde.dim]),
                                            dtype=tf.keras.backend.floatx())
        else:
            n = 0
            # # # Y
            filename = 'Y_' + str(n) + name_convention + '.index'  # standard tf format conversion
            print(filename)
            print(os.listdir(path))
            if filename not in os.listdir(path):
                warnings.warn('Y[%u] has not been found in folder' % n)
            else:
                self.Y[n] = SubNet(self.model_y_0.Yn.config, self.model_y_0.Yn.outputshape)
                self.Y[n](dummy_input, True)
                logging.info("@load_from_file: Y: signatures reinitialized")
                self.Y[n].load_weights(path + '/Y_' + str(n) + name_convention)
                self.Y[n].trainable = False
                logging.info("@load_from_file: Y: reloaded model frozen")
                # self.Y[n] = tf.keras.models.load_model(path + '/Y_' + str(n), custom_objects={'SubNet': SubNet})

            # # # Z
            filename = 'Z_' + str(n) + name_convention + '.index'  # standard tf format conversion
            if filename not in os.listdir(path):
                warnings.warn('Z[%u] has not been found in folder' % n)
            else:
                if self.model_z_0 is None:
                    self.Z[n] = SubNet(self.model_y_0.Zn.config, self.model_y_0.Zn.outputshape)
                else:
                    self.Z[n] = SubNet(self.model_z_0.Zn.config, self.model_z_0.Zn.outputshape)
                self.Z[n](dummy_input, True)
                logging.info("@load: Z: signatures reinitialized")
                self.Z[n].load_weights(path + '/Z_' + str(n) + name_convention)
                self.Z[n].trainable = False
                logging.info("@load_from_file: Z: reloaded model frozen")
                # self.Z[n] = tf.keras.models.load_model(path + '/Z_' + str(n), custom_objects={'SubNet': SubNet})

            # # # Gamma
            if not self.is_gamma_autodiff:
                filename = 'G_' + str(n) + name_convention + '.index'  # standard tf format conversion
                if filename not in os.listdir(path):
                    warnings.warn('Gamma[%u] has not been found in folder' % n)
                else:
                    self.G[n] = SubNet(self.model_z_0.Gn.config, self.model_z_0.Gn.outputshape)
                    self.G[n](dummy_input, True)
                    logging.info("@load: G: signatures reinitialized")
                    self.G[n].load_weights(path + '/G_' + str(n) + name_convention)
                    self.G[n].trainable = False
                    logging.info("@load_from_file: Z: reloaded model frozen")
                    # self.Z[n] = tf.keras.models.load_model(path + '/Z_' + str(n), custom_objects={'SubNet': SubNet})


        # # n = 1, ..., N-1
        for n in range(0, len(self.t) - 1):
            # # # Y
            filename = 'Y_' + str(n) + name_convention + '.index'  # standard tf format conversion
            print(filename)
            print(os.listdir(path))
            if filename not in os.listdir(path):
                warnings.warn('Y[%u] has not been found in folder' % n)
            else:
                self.Y[n] = SubNet(self.model_y_n.Yn.config, self.model_y_n.Yn.outputshape)
                self.Y[n](dummy_input, True)
                logging.info("@load_from_file: Y: signatures reinitialized")
                self.Y[n].load_weights(path + '/Y_' + str(n) + name_convention)
                self.Y[n].trainable = False
                logging.info("@load_from_file: Y: reloaded model frozen")
                # self.Y[n] = tf.keras.models.load_model(path + '/Y_' + str(n), custom_objects={'SubNet': SubNet})

            # # # Z
            filename = 'Z_' + str(n) + name_convention + '.index'  # standard tf format conversion
            if filename not in os.listdir(path):
                warnings.warn('Z[%u] has not been found in folder' % n)
            else:
                if self.model_z_n is not None:
                    self.Z[n] = SubNet(self.model_z_n.Zn.config, self.model_z_n.Zn.outputshape)
                else:
                    self.Z[n] = SubNet(self.model_y_n.Zn.config, self.model_y_n.Zn.outputshape)
                self.Z[n](dummy_input, True)
                logging.info("@load: Z: signatures reinitialized")
                self.Z[n].load_weights(path + '/Z_' + str(n) + name_convention)
                self.Z[n].trainable = False
                logging.info("@load_from_file: Z: reloaded model frozen")
                # self.Z[n] = tf.keras.models.load_model(path + '/Z_' + str(n), custom_objects={'SubNet': SubNet})

            # # # Gamma
            if not self.is_gamma_autodiff:
                filename = 'G_' + str(n) + name_convention + '.index'  # standard tf format conversion
                if filename not in os.listdir(path):
                    warnings.warn('G[%u] has not been found in folder' % n)
                else:
                    self.G[n] = SubNet(self.model_z_n.Gn.config, self.model_z_n.Gn.outputshape)
                    self.G[n](dummy_input, True)
                    logging.info("@load: G: signatures reinitialized")
                    self.G[n].load_weights(path + '/G_' + str(n) + name_convention)
                    self.G[n].trainable = False
                    logging.info("@load_from_file: G: reloaded model frozen")
                    # self.Z[n] = tf.keras.models.load_model(path + '/Z_' + str(n), custom_objects={'SubNet': SubNet})


        # # n = N
        self.Y[-1] = lambda x, training: self.bsde.g_tf(self.bsde.T, x)
        self.Z[-1] = lambda x, training: tf.einsum("MJd, Mdn -> MJn",
                                                   self.bsde.gradx_g_tf(self.bsde.T, x),
                                                   self.bsde.sigma_process_tf(self.bsde.T, x))

        logging.info('Nets loaded from files')
        return 0

    def generate_time_step_report(self, path, n, hist, fig_sample=None, M=2**10, markersize=3,
                                  name_convention='_weights'):
        if fig_sample is None:
            sample = self.bsde.sample(M)
        else:
            sample = fig_sample


        # Plotting
        if self.bsde.m == 1:
            if self.net_config.is_plot:
                self.plot_Yn(n, sample, path + '/plots', markersize)
                self.plot_Zn(n, sample, path + '/plots', markersize)
                self.plot_Gn(n, sample, path + '/plots', markersize)
                logging.info('Plots saved to files.')
        else:
            logging.info("@!!Multi-D BSDE encountered: no plotting")

        # Training Histories
        # hist is a dictionary!
        self.save_step_history(n, hist, path + '/train_histories')
        logging.info('Histories saved to files.')

        # save models
        self.save_models_n(n, path + '/trained_nets', name_convention)
        logging.info('Histories saved to files.')

        return 0

    def plotter_pathwise(self, path, is_gamma=True, M=4, markersize=3):
        if not self.is_trained:
            raise ValueError('Model not trained: pathwise trajectories not accesible')
        if not self.bsde.is_sigma_in_control:
            warnings.warn('The BW equation is formulated as sigma * Z * dW => u_prime will be multiplied with sigma',
                          RuntimeWarning)
            pass
        if not os.path.exists(path):
            os.makedirs(path)
        dw_sample, x_sample = self.bsde.sample(M)
        if is_gamma:
            x, y, z, gamma = self.call_pathwise(x_sample, is_gamma)
            x, y_anal, z_anal, gamma_anal = self.call_pathwise_analytical(x, is_gamma)
        else:
            x, y, z = self.call_pathwise(x_sample, is_gamma)
            x, y_anal, z_anal = self.call_pathwise_analytical(x, is_gamma)

        # Y plot
        fig, ax = plt.subplots(figsize=(8, 4))
        annotation = 'T=%.2e\nN=%u' % (self.bsde.T, self.bsde.N)
        for i in range(M):
            y_t = y[i, :, :].T
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.t, y_t, label=('' if i == 0 else '_') + '$Y$, path=%u' % i, color=color, marker='x')
            if self.bsde.is_y_theoretical:
                y_anal_t = y_anal[i, :, :].T
                ax.plot(self.t, y_anal_t, linestyle='dashed',
                        label=('' if i == 0 else '_') + 'analytical, path=%u' % i, color=color)
        if self.bsde.is_y_theoretical:
            if y_anal_t[0, 0] == 0:
                # zero solution, relative error does not say anything, rather report absolute
                logging.info("@pathwise plotter: zero solution encountered, reported error is absolute not relative")
                l2err = np.linalg.norm(y_anal_t[0, :] - y_t[0, :])
                annotation += '\nY0-Error(abs)=%.2e' % l2err
            else:
                l2err = np.linalg.norm(y_anal_t[0, :] - y_t[0, :]) / np.linalg.norm(y_anal_t[0, :])
                annotation += '\nY0-Error(rel)=%.2e' % l2err
        plt.xlabel('$t$')
        plt.ylabel('$Y_t$')
        plt.title('$Y$\nBSDE solutions pathwise')
        ax.annotate(annotation,
                    (1, 0),
                    xytext=(4, -4),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    fontweight='bold',
                    color='white',
                    backgroundcolor='red',
                    ha='left', va='top'
                    )
        plt.grid()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        fig.savefig(path + '/Y_pathwise.png', format='png')
        plt.close(fig)
        if self.bsde.dim > 1:
            warnings.warn('multi d plotting not yet implemented: will only plot y')
            return 1
        # Z plot
        fig, ax = plt.subplots(figsize=(8, 4))
        annotation = 'T=%.2e\nN=%u' % (self.bsde.T, self.bsde.N)
        for i in range(M):
            z_t = z[i, :, :].T
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.t, z_t, label=('' if i == 0 else '_') + '$Z$, path=%u' % i, color=color, marker='x')
            if self.bsde.is_y_theoretical:
                z_anal_t = z_anal[i, :, :].T
                ax.plot(self.t, z_anal_t, linestyle='dashed',
                        label=('' if i == 0 else '_') + 'analytical, path=%u' % i,
                        color=color)
        if self.bsde.is_y_theoretical:
            if z_anal_t[0, 0] == 0:
                logging.info("@pathwise plotter: zero solution encountered, reported error is absolute not relative")
                l2err = np.linalg.norm(z_anal_t[0, :] - z_t[0, :])
                annotation += '\nZ0-Error(abs)=%.2e' % l2err
            else:
                l2err = np.linalg.norm(z_anal_t[0, :] - z_t[0, :]) / np.linalg.norm(z_anal_t[0, :])
                annotation += '\nZ0-Error=%.2e(rel)' % l2err
        plt.xlabel('$t$')
        plt.ylabel('$Z_t$')
        plt.title('$Z$\nBSDE solutions pathwise')
        ax.annotate(annotation,
                    (1, 0),
                    xytext=(4, -4),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    fontweight='bold',
                    color='white',
                    backgroundcolor='red',
                    ha='left', va='top'
                    )
        plt.grid()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        fig.savefig(path + '/Z_pathwise.png', format='png')
        plt.close(fig)

        if is_gamma:
            fig, ax = plt.subplots(figsize=(8, 4))
            annotation = 'T=%.2e\nN=%u' % (self.bsde.T, self.bsde.N)
            for i in range(M):
                gamma_t = gamma[i, :, :, :].T
                color = next(ax._get_lines.prop_cycler)['color']
                ax.plot(self.t, gamma_t.flatten(), label=('' if i == 0 else '_') + '$\Gamma$, path=%u' % i, color=color, marker='x')
                if self.bsde.is_y_theoretical:
                    gamma_anal_t = gamma_anal[i, :, :, :].T
                    ax.plot(self.t, gamma_anal_t.flatten(), linestyle='dashed',
                            label=('' if i == 0 else '_') + 'analytical, path=%u' % i,
                            color=color)
            if self.bsde.is_y_theoretical:
                if gamma_anal_t[0, 0, 0] == 0:
                    logging.info(
                        "@pathwise plotter: zero solution encountered, reported error is absolute not relative")
                    l2err = np.linalg.norm(gamma_anal_t[0, :, :] - gamma_t[0, :, :], axis=(0, 1), ord=2)
                    annotation += '\nGamma0-Error(abs)=%.2e' % l2err
                else:
                    l2err = np.linalg.norm(gamma_anal_t[0, :, :] - gamma_t[0, :, :], axis=(0, 1), ord=2) / np.linalg.norm(gamma_anal_t[0, :, :], axis=(0, 1), ord=2)
                    annotation += '\nGamma0-Error=%.2e(rel)' % l2err
            plt.xlabel('$t$')
            plt.ylabel('$\Gamma_t$')
            plt.title('$\Gamma$\nBSDE solutions pathwise')
            ax.annotate(annotation,
                        (1, 0),
                        xytext=(4, -4),
                        xycoords='axes fraction',
                        textcoords='offset points',
                        fontweight='bold',
                        color='white',
                        backgroundcolor='red',
                        ha='left', va='top'
                        )
            plt.grid()
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            fig.savefig(path + '/Gamma_pathwise.png', format='png')
            plt.close(fig)

        return 0

    def call_pathwise(self, x, is_gamma=False):
        # # # x is the trajectories of the forward diffusion over the whole time window
        M, d, Np1 = x.shape
        y = np.zeros(shape=[M, 1, Np1], dtype=tf.keras.backend.floatx())
        z = np.zeros(shape=x.shape, dtype=tf.keras.backend.floatx())

        if is_gamma:
            gamma = np.zeros(shape=[M, d, d, Np1], dtype=tf.keras.backend.floatx())

        # n = 0
        if not self.is_nn_at_t0:
            all_one_vec = tf.ones(shape=[M, 1], dtype=tf.keras.backend.floatx())
            if self.Y[0] is not None:
                y[:, :, 0] = (all_one_vec * self.Y[0]).numpy()
            else:
                y[:, :, 0] = np.nan

            if self.Z[0] is not None:
                z[:, :, 0] = tf.matmul(all_one_vec, self.Z[0]).numpy()
            else:
                z[:, :, 0] = np.nan

            if is_gamma:
                # if self.G[0] is not None:
                if self.G[0] is not None:
                    gamma[:, :, :, 0] = tf.einsum('Mi, ijk -> Mjk', all_one_vec, self.G[0])
                else:
                    gamma[:, :, :, 0] = np.nan
        else:
            if self.Y[0] is not None:
                y[:, :, 0] = self.Y[0](x[:, :, 0], False).numpy()
            else:
                y[:, :, 0] = np.nan
            if self.Z[0] is not None:
                z[:, :, 0] = self.Z[0](x[:, :, 0], False).numpy()
            else:
                z[:, :, 0] = np.nan
            if is_gamma:
                if self.is_gamma_autodiff:
                    if self.Z[0] is not None:
                        if self.is_z_autodiff:
                            x_tf = tf.constant(x[:, :, 0])
                            with tf.GradientTape() as tape:
                                tape.watch(x_tf)
                                sigma_tf = self.bsde.ForwardDiffusion.sigma_process_tf(self.t[0], x_tf)
                                z_tf = tf.einsum("Mij, Mj-> Mi",
                                                 sigma_tf,
                                                 self.Y[0].grad_call(x_tf, False))
                            gamma[:, :, :, 0] = tape.batch_jacobian(z_tf, x_tf).numpy()
                            del z_tf
                        else:
                            gamma[:, :, :, 0] = self.Z[0].jacobian_call(x[:, :, 0], False).numpy()
                    else:
                        gamma[:, :, :, 0] = np.nan
                else:
                    if self.G[0] is not None:
                        gamma[:, :, :, 0] = self.G[0](x[:, :, 0], False).numpy()
                    else:
                        gamma[:, :, :, 0] = np.nan

        # n = 1, ..., N-1
        for n in range(1, self.bsde.N):
            x_n = x[:, :, n]
            if self.Y[n] is not None:
                y[:, :, n] = self.Y[n](x_n, False).numpy()
                if not self.is_z_autodiff:
                    z[:, :, n] = self.Z[n](x_n, False).numpy()
                    if is_gamma:
                        if self.is_gamma_autodiff:
                            gamma[:, :, :, n] = self.Z[n].jacobian_call(x_n, False).numpy()
                            # gamma[:, :, :, n] = 0
                        else:
                            gamma[:, :, :, n] = self.G[n](x_n, False).numpy()

                else:
                    grad_y_n = self.Y[n].grad_call(x_n, False).numpy()
                    sigma_n = self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.t[n], x_n).numpy()
                    z[:, :, n] = np.einsum('aij, aj -> ai', sigma_n, grad_y_n)
                    if is_gamma:
                        if self.bsde.ForwardDiffusion.name != 'ABM':
                            x_tf = tf.constant(x_n)
                            with tf.GradientTape() as tape:
                                tape.watch(x_tf)
                                sigma_tf = self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.t[n], x_tf)
                                grad_y_tf = self.Y[n].grad_call(x_n, False)
                                z_tf = tf.einsum("Mij, Mj -> Mi", sigma_tf, grad_y_tf)

                            gamma[:, :, :, n] = tape.batch_jacobian(z_tf, x_tf).numpy()
                            del z_tf, grad_y_tf, sigma_tf
                        else:
                            hess_y_n = self.Y[n].hessian_call(x_n, False).numpy()
                            gamma[:, :, :, n] = self.bsde.ForwardDiffusion.sigma * hess_y_n
            else:
                y[:, :, n] = np.nan
                z[:, :, n] = np.nan


        y[:, :, -1] = self.bsde.g_tf(self.bsde.T, x[:, :, -1]).numpy()
        if not self.bsde.is_sigma_in_control:
            z[:, :, -1] = tf.einsum('ink,ik->in',
                                    self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.T, tf.constant(x[:, :, -1])),
                                    self.bsde.gradx_g_tf(self.bsde.T, tf.constant(x[:, :, -1]))).numpy()
            if is_gamma:
                if self.bsde.ForwardDiffusion.name != 'ABM':
                    # x_tf = tf.constant(x[:, :, -1])
                    # nabla_sigma = self.bsde.ForwardDiffusion.nabla_sigma_process_tf(self.bsde.total_time, x_tf)
                    # sigma = self.bsde.ForwardDiffusion.sigma_process_tf(self.bsde.total_time, x_tf)
                    # with tf.GradientTape() as tape:
                    #     grad_g = self.bsde.gradx_g_tf(self.bsde.total_time, x_tf)
                    # hess_g = tape.batch_jacobian(grad_g, x_tf)
                    # gamma[:, :, :, -1] = tf.einsum("Mijk, Mk -> Mij", nabla_sigma, grad_g).numpy()\
                    #                      +tf.einsum("Mij, Mjk -> Mik", sigma, hess_g).numpy()
                    gamma[:, :, :, -1] = self.bsde.gamma_analytical(self.bsde.T, x[:, :, -1])
                else:
                    x_tf = tf.constant(x[:, :, -1])
                    with tf.GradientTape() as tape:
                        grad_g = self.bsde.gradx_g_tf(self.bsde.T, x_tf)
                    hess_g = tape.batch_jacobian(grad_g, x_tf)
                    gamma[:, :, :, -1] = self.bsde.ForwardDiffusion.sigma * hess_g.numpy()

                    gamma[:, :, :, -1] = self.bsde.gamma_analytical(self.bsde.T, x[:, :, -1])
        else:
            z[:, :, -1] = self.bsde.gradx_g_tf(self.bsde.T, x[:, :, -1]).numpy()
            if is_gamma:
                gamma[:, :, :, -1] = self.bsde.hessx_g_tf(self.bsde.T, x[:, :, -1]).numpy()

        if is_gamma:
            return x, y, z, gamma

        return x, y, z

    def call_pathwise_analytical(self, x, is_gamma=False):
        M, d, Np1 = x.shape
        y = np.zeros(shape=[M, 1, Np1], dtype=tf.keras.backend.floatx())
        z = np.zeros(shape=x.shape, dtype=tf.keras.backend.floatx())
        if is_gamma:
            gamma = np.zeros(shape=[M, d, d, Np1], dtype=tf.keras.backend.floatx())


        for n in range(self.bsde.N + 1):
            t_n = self.bsde.t[n]
            x_n = x[:, :, n]
            y[:, :, n] = self.bsde.y_analytical(t_n, x_n)
            z[:, :, n] = self.bsde.z_analytical(t_n, x_n)
            if is_gamma:
                gamma[:, :, :, n] = self.bsde.gamma_analytical(t_n, x_n)

        if is_gamma:
            return x, y, z, gamma
        return x, y, z

    def find_exercise_index(self, x_sample, N_rebalance=None):
        M, d, N_finest_plus1 = tf.shape(x_sample)

        N_finest = N_finest_plus1 - 1
        if N_rebalance is None:
            N_rebalance = N_finest
        if N_finest != self.bsde.N:
            raise ValueError
        if (N_finest % N_rebalance) != 0:
            raise ValueError
        mod = int(N_finest / N_rebalance)

        exercise_index = N_finest * tf.ones([M, self.bsde.J], dtype="int32")
        # mj: contains the time index of the date when the j'th option is exercised on the m'th path
        for n in range(N_rebalance + 1):
            n_subidx = int(n * mod)

            t_n = self.t[n_subidx]
            x_n = x_sample[..., n_subidx]

            y_tilde_n = self.Y[n](x_n, training=False)
            l_n = self.bsde.g_tf(t_n, x_n)
            is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_n > y_tilde_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n_subidx])

            exercise_index = tf.where(is_exercised == 1.0,
                                      tf.where(n_subidx < exercise_index, n_subidx, exercise_index),
                                      exercise_index)

        return exercise_index

    def delta_hedging_black_scholes_alpha_n(self, n, x_n, is_z_tilde, exercise_index):
        # exercise_index: helps to eliminate options which have been exercised before time t_n and thus are no longer part of the replicating portfolio
        t_n = self.t[n]
        if self.bsde.m != self.bsde.d:
            raise ValueError

        is_held_at_n = tf.where(exercise_index < n, 0.0, 1.0)

        y_n = self.Y[n](x_n, training=False)
        z_n = self.Z[n](x_n, training=False)

        l_n = self.bsde.g_tf(t_n, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_n > y_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n])
        y_n = y_n + is_exercised * (l_n - y_n)
        if is_z_tilde:
            grad_l = self.bsde.gradx_g_tf(t_n, x_n)
            exercised_z_n = tf.einsum("Mjm, Mmn -> Mjn", grad_l, self.bsde.sigma_process_tf(t_n, x_n))
            z_n = z_n + tf.einsum("MJ, MJd -> MJd", is_exercised, exercised_z_n - z_n)
        else:
            pass

        inv_sigma_n = self.bsde.inverse_sigma_process_tf(t_n, x_n)

        # alpha_h_n = np.sum(np.einsum("MJd, Mdi -> MJi", z_n, inv_sigma_n[..., 0: self.bsde.m]), axis=1)
        alpha_h_n = np.einsum("MJ, MJi -> Mi", is_held_at_n, np.einsum("MJd, Mdi -> MJi", z_n, inv_sigma_n[..., 0: self.bsde.m]))

        return alpha_h_n, y_n


    def delta_hedging_black_scholes_portfolio(self, x_finest_sample, dir_hedging_weights, N_rebalance, exercise_index_rebalance):
        M, d, N_finest_plus_1 = x_finest_sample.shape
        N_finest = N_finest_plus_1 - 1
        if N_finest != self.bsde.N:
            raise ValueError
        if (N_finest % N_rebalance) != 0:
            raise ValueError

        mod = N_finest / N_rebalance

        portfolio_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        t_rebalance = []
        for n in range(0, N_rebalance + 1):
            n_subidx = int(n * mod)

            t_n = self.bsde.t[n_subidx]
            t_rebalance.append(t_n)
            s_n = x_finest_sample[:, 0: self.bsde.m, n_subidx]
            alpha_h_n = np.load(dir_hedging_weights + "/alpha_h_n_" + str(n) + ".npy")
            y_n = np.load(dir_hedging_weights + "/y_n_" + str(n) + ".npy")

            is_held_at_n = tf.where(exercise_index_rebalance <= n_subidx, 0.0, 1.0)

            is_giving_payoff_at_n = tf.where(exercise_index_rebalance == n_subidx, 1.0, 0.0)

            if n == 0:
                # hedging
                b_n = tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True)\
                      - tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True)
                y_0 = 1 * y_n
            else:

                # hedging
                b_n = np.exp(self.bsde.r * (t_n - t_prev)) * b_n \
                      - tf.reduce_sum(s_n * (alpha_h_n - alpha_h_prev), axis=-1, keepdims=True)

                dividend_paid = self.bsde.q * tf.reduce_sum(alpha_h_prev * s_prev, axis=-1, keepdims=True) * (
                        t_n - t_prev)
                b_n += dividend_paid

                b_n += -tf.reduce_sum(is_giving_payoff_at_n * y_n, axis=1, keepdims=True)  # instant payoff

                portfolio_sample[..., n] = -tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True)\
                                           + tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True) + b_n


            t_prev = 1 * t_n
            b_prev = 1 * b_n
            y_prev = 1 * y_n
            s_prev = 1 * s_n
            alpha_h_prev = 1 * alpha_h_n

        t_rebalance = np.array(t_rebalance)
        pnl_rel_sample = np.einsum("Mij, Mi -> Mij",
                                   np.einsum("N, MiN -> MiN", np.exp(-self.bsde.r * t_rebalance), portfolio_sample),
                                   np.sum(y_0, axis=1, keepdims=True) ** (-1))



        return portfolio_sample, pnl_rel_sample

    def gamma_hedging_black_scholes_y_nabla_hess_n(self, n, x_n, is_z_tilde, exercise_index):
        t_n = self.t[n]
        if self.bsde.m != self.bsde.d:
            raise ValueError

        is_held_at_n = tf.where(exercise_index < n, 0.0, 1.0)

        M, d = x_n.shape

        t0 = time.time()
        t_n = self.t[n]

        y_n = self.Y[n](x_n, training=False)
        z_n = self.Z[n](x_n, training=False)
        if is_z_tilde:
            if n < self.bsde.N:
                # print(n)
                gamma_n = self.G[n](x_n, training=False)
        else:
            x_tf = tf.constant((x_n))
            with tf.GradientTape() as tape:
                tape.watch(x_tf)
                z_n = self.Z[n](x_tf, False)
            gamma_n = tape.batch_jacobian(z_n, x_tf)



        l_n = self.bsde.g_tf(t_n, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_n > y_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n])
        y_n = y_n + is_exercised * (l_n - y_n)
        if is_z_tilde:
            grad_l = self.bsde.gradx_g_tf(t_n, x_n)
            exercised_z_n = tf.einsum("Mjm, Mmn -> Mjn", grad_l, self.bsde.sigma_process_tf(t_n, x_n))
            z_n = z_n + tf.einsum("MJ, MJd -> MJd", is_exercised, exercised_z_n - z_n)

        nabla_y_n = self.bsde.get_gradient_from_z(t_n, x_n, z_n, is_held_at_n)
        if n < self.bsde.N:
            # hess_y_n = 1 * gamma_n.numpy()
            hess_y_n = self.bsde.get_hessian_from_gamma(t_n, x_n, nabla_y_n, gamma_n, is_held_at_n)  # Gamma = \nabla Z = sigma^T Hess u + nabla u nabla sigma
            # for i in range(d):
            #     nabla_sigma_ith_row = self.bsde.nabla_sigma_process_jth_row_tf(t_n, tf.constant(x_n), i)
            #     hess_y_n[:, :, i] -= np.einsum("MJd, Mdq -> MJq", nabla_y_n, nabla_sigma_ith_row.numpy())
            # hess_y_n = np.einsum("MJdm, Mmk -> MJdk", hess_y_n, inv_sigma_n)
        else:
            hess_y_n = tf.einsum("MJ, MJdq -> Mdq", is_held_at_n, self.bsde.hessx_g_tf(t_n, x_n)).numpy()
        t_grads = time.time()

        return y_n, nabla_y_n, hess_y_n


    def gamma_hedging_black_scholes_portfolio(self, x_finest_sample, dir_sec, dir_hedging_weights, N_rebalance,
                                              exercise_index_rebalance):
        M, d, N_finest_plus_1 = x_finest_sample.shape
        N_finest = N_finest_plus_1 - 1
        if N_finest != self.bsde.N:
            raise ValueError
        if (N_finest % N_rebalance) != 0:
            raise ValueError

        mod = N_finest / N_rebalance

        portfolio_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        t_rebalance = []
        for n in range(0, N_rebalance + 1):
            n_subidx = int(n * mod)

            t_n = self.bsde.t[n_subidx]
            t_rebalance.append(t_n)
            s_n = x_finest_sample[:, 0: self.bsde.m, n_subidx]
            I_ij_n = np.load(dir_sec + "/sec_sample_n_" + str(n_subidx) + ".npy")
            # I_ij_n = sec_finest_sample[..., n]
            alpha_h_n = np.load(dir_hedging_weights + "/alpha_h_n_" + str(n) + ".npy")
            beta_h_n = np.load(dir_hedging_weights + "/beta_h_n_" + str(n) + ".npy")
            y_n = np.load(dir_hedging_weights + "/y_n_" + str(n) + ".npy")

            is_held_at_n = tf.where(exercise_index_rebalance <= n_subidx, 0.0, 1.0)
            is_giving_payoff_at_n = tf.where(exercise_index_rebalance == n_subidx, 1.0, 0.0)

            if n == 0:
                # hedging
                b_n = tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True)\
                      - tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True) \
                      - tf.reduce_sum(beta_h_n * I_ij_n, axis=-1, keepdims=True)
                y_0 = 1 * y_n

            else:
                # hedging
                b_n = np.exp(self.bsde.r * (t_n - t_prev)) * b_n \
                      - tf.reduce_sum(s_n * (alpha_h_n - alpha_h_prev), axis=-1, keepdims=True) \
                      - tf.reduce_sum(I_ij_n * (beta_h_n - beta_h_prev), axis=-1, keepdims=True)

                dividend_paid = self.bsde.q * tf.reduce_sum(alpha_h_prev * s_prev, axis=-1, keepdims=True) * (t_n - t_prev)
                b_n += dividend_paid

                b_n += -tf.reduce_sum(is_giving_payoff_at_n * y_n, axis=1, keepdims=True)  # instant payoff

                portfolio_sample[..., n] = -tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True)\
                                           + tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True) \
                                           + tf.reduce_sum(beta_h_n * I_ij_n, axis=-1, keepdims=True) \
                                           + b_n

            t_prev = t_n
            s_prev = 1 * s_n
            alpha_h_prev = 1 * alpha_h_n
            beta_h_prev = 1 * beta_h_n

        t_rebalance = np.array(t_rebalance)
        pnl_rel_sample = np.einsum("Mij, Mi -> Mij",
                                   np.einsum("N, MiN -> MiN", np.exp(-self.bsde.r * t_rebalance), portfolio_sample),
                                   np.sum(y_0, axis=1, keepdims=True) ** (-1))


        return portfolio_sample, pnl_rel_sample

    def vega_hedging_portfolio(self, x_finest_sample, sec_trajectory, diff_sec_trajectory, N_rebalance,
                               exercise_index_rebalance, is_z_tilde=True, output_dir=None, mode="vega"):
        M, d, N_finest_plus_1 = x_finest_sample.shape
        N_finest = N_finest_plus_1 - 1
        if N_finest != self.bsde.N:
            raise ValueError
        if (N_finest % N_rebalance) != 0:
            raise ValueError

        K = sec_trajectory.shape[1]

        mod = N_finest / N_rebalance

        portfolio_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        t_rebalance = []
        for n in range(0, N_rebalance + 1):
            n_subidx = int(n * mod)

            t_n = self.bsde.t[n_subidx]
            t_rebalance.append(t_n)
            x_n = x_finest_sample[..., n_subidx]
            s_n = x_n[:, 0: self.bsde.m]
            c_n = self.Y[n_subidx](x_n, training=False)
            z_tilde_n = self.Z[n_subidx](x_n, training=False)

            is_held_at_n = tf.where(exercise_index_rebalance <= n_subidx, 0.0, 1.0)
            is_giving_payoff_at_n = tf.where(exercise_index_rebalance == n_subidx, 1.0, 0.0)

            l_n = self.bsde.g_tf(t_n, x_n)
            is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_n > c_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n_subidx])
            y_n = c_n + is_exercised * (l_n - c_n)
            if is_z_tilde:
                grad_l = self.bsde.gradx_g_tf(t_n, x_n)
                exercised_z_n = tf.einsum("Mjm, Mmn -> Mjn", grad_l, self.bsde.sigma_process_tf(t_n, x_n))
                z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, exercised_z_n - z_tilde_n)
            else:
                z_n = 1 * z_tilde_n
            inv_sigma_n = self.bsde.inverse_sigma_process_tf(t_n, x_n)
            nabla_y_n = np.sum(np.einsum("MJd, Mdi -> MJi", z_n, inv_sigma_n), axis=1)


            # # # get beta ---------------------------------------------------------------------------------------------
            u_n = sec_trajectory[..., n_subidx]  # M x num_sec
            diff_u_n = diff_sec_trajectory[..., n_subidx]  # M x d x num_sec

            du_dnu_n = diff_u_n[:, self.bsde.m:, :]  # partial derivatives wrt volatility
            du_ds_n = diff_u_n[:, 0: self.bsde.m, :]

            coeff_mtx = du_dnu_n.reshape((M, self.bsde.d - self.bsde.m, K))
            rhs_vec = nabla_y_n[:, self.bsde.m:]

            pinv_coeff_mtx = np.linalg.pinv(coeff_mtx)
            beta_h_n = np.einsum("Mij, Mj -> Mi", pinv_coeff_mtx, rhs_vec)
            # ----------------------------------------------------------------------------------------------------------
            if mode == 'delta':
                beta_h_n = 0 * beta_h_n  # no hedging instrument held
            alpha_h_n = nabla_y_n[:, 0: self.bsde.m] - np.einsum("MK, MmK -> Mm", beta_h_n, du_ds_n)

            if output_dir is not None:
                np.save(output_dir + '_alpha_' + str(n) + '.npy', alpha_h_n)
                np.save(output_dir + '_beta_' + str(n) + '.npy', beta_h_n)
            if n == 0:
                # hedging
                b_n = tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True) \
                      - tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True)\
                      - tf.reduce_sum(beta_h_n * u_n, axis=-1, keepdims=True)
                y_0 = 1 * y_n
            else:

                # hedging
                b_n = np.exp(self.bsde.r * (t_n - t_prev)) * b_n \
                      - tf.reduce_sum(s_n * (alpha_h_n - alpha_h_prev), axis=-1, keepdims=True)\
                      - tf.reduce_sum(u_n * (beta_h_n - beta_h_prev), axis=-1, keepdims=True)

                dividend_paid = self.bsde.q * tf.reduce_sum(alpha_h_prev * s_prev, axis=-1, keepdims=True) * (
                        t_n - t_prev)
                b_n += dividend_paid

                b_n += -tf.reduce_sum(is_giving_payoff_at_n * y_n, axis=1, keepdims=True)  # instant payoff

                portfolio_sample[..., n] = -tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True) \
                                           + tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True) \
                                           + tf.reduce_sum(beta_h_n * u_n, axis=-1, keepdims=True) + b_n

            t_prev = 1 * t_n
            b_prev = 1 * b_n
            y_prev = 1 * y_n
            s_prev = 1 * s_n
            alpha_h_prev = 1 * alpha_h_n
            beta_h_prev = 1 * beta_h_n

        t_rebalance = np.array(t_rebalance)
        pnl_rel_sample = np.einsum("Mij, Mi -> Mij",
                                   np.einsum("N, MiN -> MiN", np.exp(-self.bsde.r * t_rebalance), portfolio_sample),
                                   np.sum(y_0, axis=1, keepdims=True) ** (-1))

        return portfolio_sample, pnl_rel_sample

    def vomma_hedging_portfolio(self, x_finest_sample, sec_trajectory, diff_sec_trajectory, hess_sec_trajectory,
                                N_rebalance, exercise_index_rebalance, is_z_tilde=True, output_dir=None, mode="upper_triangular"):
        M, d, N_finest_plus_1 = x_finest_sample.shape
        N_finest = N_finest_plus_1 - 1
        if N_finest != self.bsde.N:
            raise ValueError
        if (N_finest % N_rebalance) != 0:
            raise ValueError

        if mode == "upper_triangular":
            L = int(d * (d + 1) / 2)
        elif mode == "upper_triangular_gamma":
            m = int(d / 2)
            L = int(m * (m + 1) / 2)
        elif mode == "diagonal":
            L = d
        elif mode == "full_matrix":
            L = d ** 2
        else:
            raise NotImplementedError
        K = sec_trajectory.shape[1]

        mod = N_finest / N_rebalance

        portfolio_sample = np.zeros(shape=[M, 1, N_rebalance + 1], dtype=tf.keras.backend.floatx())
        t_rebalance = []
        for n in range(0, N_rebalance + 1):
            n_subidx = int(n * mod)

            t_n = self.bsde.t[n_subidx]
            t_rebalance.append(t_n)
            x_n = x_finest_sample[..., n_subidx]
            s_n = x_n[:, 0: self.bsde.m]
            c_n = self.Y[n_subidx](x_n, training=False)
            z_tilde_n = self.Z[n_subidx](x_n, training=False)
            if n_subidx < self.bsde.N:
                if is_z_tilde:
                    gamma_n = self.G[n_subidx](x_n, training=False)
                else:
                    x_tf = tf.constant((x_n))
                    with tf.GradientTape() as tape:
                        tape.watch(x_tf)
                        z_n = self.Z[n_subidx](x_tf, False)
                    gamma_n = tape.batch_jacobian(z_n, x_tf)
            is_held_at_n = tf.where(exercise_index_rebalance <= n_subidx, 0.0, 1.0)
            is_giving_payoff_at_n = tf.where(exercise_index_rebalance == n_subidx, 1.0, 0.0)

            l_n = self.bsde.g_tf(t_n, x_n)
            is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_n > c_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n_subidx])
            y_n = c_n + is_exercised * (l_n - c_n)
            if is_z_tilde:
                grad_l = self.bsde.gradx_g_tf(t_n, x_n)
                exercised_z_n = tf.einsum("Mjm, Mmn -> Mjn", grad_l, self.bsde.sigma_process_tf(t_n, x_n))
                z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, exercised_z_n - z_tilde_n)
            else:
                z_n = 1 * z_tilde_n
            inv_sigma_n = self.bsde.inverse_sigma_process_tf(t_n, x_n)
            nabla_y_n = np.sum(np.einsum("MJd, Mdi -> MJi", z_n, inv_sigma_n), axis=1)

            if n_subidx < self.bsde.N:
                nabla_u_nabla_sigma = tf.einsum("Md, Mdqm -> Mqm",
                                                nabla_y_n, self.bsde.nabla_sigma_process_tf(t_n, x_n))
                # inv_sigmaT_n = self.bsde.inverse_sigma_transpose_process_tf(t_n, x_n)
                hess_y_n = tf.einsum("Mij, Mik -> Mjk", inv_sigma_n, tf.reduce_sum(gamma_n, axis=1) - nabla_u_nabla_sigma)
            else:
                hess_y_n = tf.reduce_sum(self.bsde.hessx_g_tf(t_n, x_n), axis=1)


            # # # get beta ---------------------------------------------------------------------------------------------
            u_n = sec_trajectory[..., n_subidx]  # M x num_sec
            diff_u_n = diff_sec_trajectory[..., n_subidx]  # M x d x num_sec
            hess_u_n = hess_sec_trajectory[..., n_subidx]  # M x d x d x num_sec

            du_dnu_n = diff_u_n[:, self.bsde.m:, :]  # partial derivatives wrt volatility
            du_ds_n = diff_u_n[:, 0: self.bsde.m, :]
            coeff_mtx = du_dnu_n.reshape((M, self.bsde.d - self.bsde.m, K))
            rhs_vec = nabla_y_n[:, self.bsde.m:]

            if mode == 'upper_triangular':
                second_order_coeff_mtx = np.zeros(shape=[M, L, K], dtype=tf.keras.backend.floatx())
                second_order_rhs = np.zeros(shape=[M, L], dtype=tf.keras.backend.floatx())
                counter = 0
                for i in range(d):
                    for j in range(i, d):
                        second_order_coeff_mtx[:, counter, :] = hess_u_n[:, i, j, :]
                        second_order_rhs[:, counter] = hess_y_n[:, i, j]
                        counter += 1
            elif mode == "upper_triangular_gamma":
                # only looking at the first mxm block corresponding to assets
                second_order_coeff_mtx = np.zeros(shape=[M, L, K], dtype=tf.keras.backend.floatx())
                second_order_rhs = np.zeros(shape=[M, L], dtype=tf.keras.backend.floatx())
                counter = 0
                for i in range(m):
                    for j in range(i, m):
                        second_order_coeff_mtx[:, counter, :] = hess_u_n[:, i, j, :]
                        second_order_rhs[:, counter] = hess_y_n[:, i, j]
                        counter += 1
            elif mode == "diagonal":
                second_order_coeff_mtx = np.zeros(shape=[M, L, K], dtype=tf.keras.backend.floatx())
                second_order_rhs = np.zeros(shape=[M, L], dtype=tf.keras.backend.floatx())
                counter = 0
                for i in range(d):
                    second_order_coeff_mtx[:, counter, :] = hess_u_n[:, i, i, :]
                    second_order_rhs[:, counter] = hess_y_n[:, i, i]
                    counter += 1
            elif mode == "full_matrix":
                # only looking at the first mxm block corresponding to assets
                second_order_coeff_mtx = np.zeros(shape=[M, L, K], dtype=tf.keras.backend.floatx())
                second_order_rhs = np.zeros(shape=[M, L], dtype=tf.keras.backend.floatx())
                counter = 0
                for i in range(d):
                    for j in range(d):
                        second_order_coeff_mtx[:, counter, :] = hess_u_n[:, i, j, :]
                        second_order_rhs[:, counter] = hess_y_n[:, i, j]
                        counter += 1
            else:
                raise NotImplementedError
            coeff_mtx = np.concatenate([coeff_mtx, second_order_coeff_mtx], axis=1)
            rhs_vec = np.concatenate([rhs_vec, second_order_rhs], axis=1)
            pinv_coeff_mtx = np.linalg.pinv(coeff_mtx)
            beta_h_n = np.einsum("Mij, Mj -> Mi", pinv_coeff_mtx, rhs_vec)
            # ----------------------------------------------------------------------------------------------------------
            residual = np.einsum("Mij, Mj -> Mi", coeff_mtx, beta_h_n) - rhs_vec
            alpha_h_n = nabla_y_n[:, 0: self.bsde.m] - np.einsum("MK, MmK -> Mm", beta_h_n, du_ds_n)

            if output_dir is not None:
                np.save(output_dir + '_alpha_' + str(n) + '.npy', alpha_h_n)
                np.save(output_dir + '_beta_' + str(n) + '.npy', beta_h_n)
            if n == 0:
                # hedging
                b_n = tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True) \
                      - tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True)\
                      - tf.reduce_sum(beta_h_n * u_n, axis=-1, keepdims=True)
                y_0 = 1 * y_n
            else:

                # hedging
                b_n = np.exp(self.bsde.r * (t_n - t_prev)) * b_n \
                      - tf.reduce_sum(s_n * (alpha_h_n - alpha_h_prev), axis=-1, keepdims=True)\
                      - tf.reduce_sum(u_n * (beta_h_n - beta_h_prev), axis=-1, keepdims=True)

                dividend_paid = self.bsde.q * tf.reduce_sum(alpha_h_prev * s_prev, axis=-1, keepdims=True) * (
                        t_n - t_prev)
                b_n += dividend_paid

                b_n += -tf.reduce_sum(is_giving_payoff_at_n * y_n, axis=1, keepdims=True)  # instant payoff

                portfolio_sample[..., n] = -tf.reduce_sum(is_held_at_n * y_n, axis=1, keepdims=True) \
                                           + tf.reduce_sum(alpha_h_n * s_n, axis=-1, keepdims=True) \
                                           + tf.reduce_sum(beta_h_n * u_n, axis=-1, keepdims=True) + b_n

            t_prev = 1 * t_n
            s_prev = 1 * s_n
            alpha_h_prev = 1 * alpha_h_n
            beta_h_prev = 1 * beta_h_n

        t_rebalance = np.array(t_rebalance)
        pnl_rel_sample = np.einsum("Mij, Mi -> Mij",
                                   np.einsum("N, MiN -> MiN", np.exp(-self.bsde.r * t_rebalance), portfolio_sample),
                                   np.sum(y_0, axis=1, keepdims=True) ** (-1))

        return portfolio_sample, pnl_rel_sample

class OneLayerBaseSolver(BackwardBSDESolverBaseClass):
    def __init__(self, config, bsde, optimizer_func=custom_adam_optimizer):
        super(OneLayerBaseSolver, self).__init__(config, bsde)
        self.model_y_n = None
        self.model_y_0 = None  # if remains None if (not self.is_nn_at_t0)

        total_iterations = self.net_config.num_iterations * self.no_of_batches
        if (self.net_config.train_size == 0) or (self.net_config.train_size is None):
            total_iterations = self.net_config.num_iterations
            logging.info("@learning_rate_schedule: assuming resampling training")
        self.optimizer_y_0 = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)
        self.optimizer_y_n = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)

    def train_minibatch(self, output_dir):
        if self.is_nn_at_t0:
            raise NotImplementedError
        start_time = time.time()
        training_history = [None] * len(self.t)  # time step many long

        train_data = self.bsde.sample(self.net_config.train_size)
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # # # start iterating over time steps
        # # N
        self.Y[-1] = lambda X_T, training: self.bsde.g_tf(self.bsde.T, X_T)  # to have the same call format as nets
        if not self.bsde.is_sigma_in_control:
            self.Z[-1] = lambda X_T, training: tf.einsum('Mi, Mij-> Mj',
                                                         self.bsde.gradx_g_tf(self.bsde.T, X_T),
                                                         self.bsde.ForwardDiffusion.sigma_process_tf(
                                                             self.bsde.T,
                                                             X_T
                                                         )
                                                         )
        else:
            self.Z[-1] = lambda X_T, training: self.bsde.gradx_g_tf(self.bsde.T, X_T)
        self.generate_time_step_report(output_dir, self.bsde.N, training_history)
        if self.is_collect_future:
            self.train_collected_future = self.initialize_collected_future(train_data)
            self.valid_collected_future = self.initialize_collected_future(valid_data)

        for n in range(self.bsde.N - 1, -1, -1):
            logging.info('''============================================================\nStarting Time Step %5u''' % n)
            if self.is_collect_future:
                self.update_collected_future(n + 1, train_data, self.train_collected_future)  # overwrites input's n+1'th idx
                self.update_collected_future(n + 1,  valid_data, self.valid_collected_future)
                logging.info('@future paths for Y, Z, gradY, gradZ updated for both train and validation sets.')

            step_history = {'Z': [], 'Y': []}
            loss = {'Z': [], 'Y': []}

            # # # Collective
            logging.info('''::::::::::::::::::::::::::\nY_%5u''' % n)
            if n < self.bsde.N - 1:
                self._reset_optimizer(self.optimizer_y_n)
                logging.info("@Yn optimizer reset to initial state.")
                # self.optimizer_y_n.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                # logging.info("@Pretrained points total epoch number reduced to %u"
                #              % self.net_config.pretrained_num_iterations)
                if (n == self.bsde.N - 2) and (n > 0):
                    # # this only needs to be done once.
                    # if self.net_config.pretrained_num_iterations < self.net_config.num_iterations:
                    #     logging.info("@interior time step's total number of iterations changed.")
                    #     logging.info("@ from: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                    #     self.optimizer_y_n.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                    #     logging.info("@ to: %d"%self.optimizer_y_n.learning_rate.total_epochs)
                    if self.net_config.pretrained_initial_learning_rate is not None:
                        self.optimizer_y_n.learning_rate = self.lr_schedule_pretrained
                        logging.info("@interior time steps learning rate schedule updated")

                    # if self.net_config.batch_size_pretrained is not None:
                    #     total_epochs_pretrained = self.net_config.pretrained_num_iterations * self.no_of_batches_pretrained
                    #     logging.info("@interior time step's batch size changed total number of iterations.")
                    #     logging.info("@ from: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                    #     self.optimizer_y_n.learning_rate.total_epochs = total_epochs_pretrained
                    #     logging.info("@ to: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                    # 
                    #     self.net_config.batch_size = 1 * self.net_config.batch_size_pretrained
                    #     logging.info("@COME BACK HERE: I OWERWROTE BATCH SIZE FOR INTERIOR TIME STEPS!!!!")

            valid_input_y_n = self.get_inputs_y(n, valid_data, self.valid_collected_future)
            train_input_y_n = self.get_inputs_y(n, train_data, self.train_collected_future)
            if n == 0:
                if self.net_config.is_t0_explicit_init:
                    self.initialize_y_0_z_0(train_data)
                    logging.info("@smart t0 average initialization")
            dataset_y_n = self.data_iterator(train_input_y_n)
            # logging.info("@Yn: number of batches: %u" % len(dataset_y_n))
            iteration_start = time.time()
            total_epochs = self.net_config.num_iterations
            if (n < self.bsde.N - 1) and (n > 0):
                total_epochs = self.net_config.pretrained_num_iterations
            for epoch in range(total_epochs + 1):
                # # # maximum number of epochs per training step
                if n == 0:
                    current_lr = self.optimizer_y_0._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                    valid_loss = self.loss_y_0(n, valid_input_y_n, False).numpy()
                    for batch in dataset_y_n:
                        self.train_step_y_0(n, batch)
                else:
                    current_lr = self.optimizer_y_n._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                    if self.is_validation_batching:
                        # since there is no automatic differentiation, this does not have to be done for time step 0
                        valid_loss = self.validation_batching(self.loss_y_n, n, valid_input_y_n, False).numpy()
                    else:
                        valid_loss = self.loss_y_n(n, valid_input_y_n, False).numpy()
                    for batch in dataset_y_n:
                        self.train_step_y_n(n, batch)
                # # # report
                elapsed_time = time.time() - start_time
                loss['Y'].append(valid_loss)
                step_history['Y'].append([epoch, loss['Y'][-1], elapsed_time])
                if self.net_config.verbose:
                    logging.info("epoch: %5u, lr: %.2e, val_loss: %.4e, elapsed time: %3u"
                                 % (epoch, current_lr, loss['Y'][-1], elapsed_time))
            iteration_end = time.time()
            dummy_input_y = train_input_y_n[0][0: 2, :]  # first element in tuple should always be x_n
            if n > 0:
                # if self.is_z_autodiff:
                #     self.Y[n],  = self.model_snapshot_y_n(dummy_input_y)
                # else:
                #     self.Y[n], self.Z[n] = self.model_snapshot_y_n(dummy_input_y)

                self.Y[n], self.Z[n] = self.model_snapshot_y_n(dummy_input_y)
            else:
                self.Y[0] = self.model_y_0.Y0
                self.Z[0] = self.model_y_0.Z0
            self.plot_Yn(n, valid_data, output_dir + '/plots')
            self.plot_Zn(n, valid_data, output_dir + '/plots')
            logging.info("@avg epoch time: %.2e"
                         % ((iteration_end - iteration_start) / (self.net_config.num_iterations + 1)))
            logging.info('''::::::::::::::::::::::::::\nCompleted Y_%5u\n::::::::::::::::::::::::::''' % n)
            # # #
            training_history[n] = step_history  # list of dictionaries!
            self.generate_time_step_report(output_dir, n, training_history)

        self.is_trained = True
        self.plotter_pathwise(output_dir)
        logging.info("@pathwise estimates saved to file, quitting training now")

        return training_history

    def train_fetch(self, output_dir):
        start_time = time.time()
        training_history = [None] * len(self.t)  # time step many long

        valid_data = self.bsde.sample(self.net_config.valid_size)

        # # # start iterating over time steps
        # # N
        self.Y[-1] = lambda X_T, training: self.bsde.g_tf(self.bsde.T,
                                                          X_T)  # to have the same call format as nets
        self.Z[-1] = lambda X_T, training: tf.einsum('MJm,Mmn->MJn',
                                                     self.bsde.gradx_g_tf(self.bsde.T, X_T),
                                                     self.bsde.sigma_process_tf(self.bsde.T,X_T)
                                                     )
        self.generate_time_step_report(output_dir, self.bsde.N, training_history)
        if self.is_collect_future:
            raise NotImplementedError

        for n in range(self.bsde.N - 1, -1, -1):
            logging.info('''============================================================\nStarting Time Step %5u''' % n)
            if self.is_collect_future:
                raise NotImplementedError

            step_history = {'Y': []}
            loss = {'Y': []}

            # # # Collective
            logging.info('''::::::::::::::::::::::::::\nY_%5u''' % n)
            if n < self.bsde.N - 1:
                self._reset_optimizer(self.optimizer_y_n)
                logging.info("@Yn optimizer reset to initial state.")
                # if self.net_config.pretrained_num_iterations < self.net_config.num_iterations:
                #     logging.info("@interior time step's total number of iterations changed.")
                #     logging.info("@ from: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                #     if n == 0:
                #         self.optimizer_y_0.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                #     else:
                #         self.optimizer_y_n.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                #     logging.info("@ to: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                if self.net_config.pretrained_initial_learning_rate is not None:
                    if n == 0:
                        self.optimizer_y_0.learning_rate = self.lr_schedule_pretrained
                    else:
                        self.optimizer_y_n.learning_rate = self.lr_schedule_pretrained
                    logging.info("@interior time steps learning rate schedule updated")

            valid_input_y_n = self.get_inputs_y(n, valid_data, self.valid_collected_future)
            if n == 0:
                if not self.is_nn_at_t0:
                    if self.net_config.is_t0_explicit_init:
                        self.initialize_y_0_z_0(self.bsde.sample(self.net_config.batch_size))
                        logging.info("@smart t0 average initialization")
            iteration_start = time.time()
            total_iterations = self.net_config.num_iterations
            if n < self.bsde.N - 1:
                total_iterations = self.net_config.pretrained_num_iterations
            for iteration in range(total_iterations + 1):
                t0_iteration = time.time()
                train_data = self.bsde.sample(self.net_config.batch_size)
                t1_sampling = time.time()
                sampling_time = t1_sampling - t0_iteration
                batch = self.get_inputs_y(n, train_data, self.valid_collected_future)
                t1_target = time.time()
                target_gathering_time = t1_target - t1_sampling
                dataset = batch
                t1_target = time.time()
                # # # maximum number of epochs per training step
                if n == 0:
                    if (iteration % self.net_config.logging_frequency) == 0:
                        try:
                            current_lr = self.optimizer_y_0.lr
                        except:
                            current_lr = self.optimizer_y_0._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                        # do not waste CPU time when it's not reported anyway
                        t0_valid_loss = time.time()
                        valid_loss = self.loss_y_0(n, valid_input_y_n, False).numpy()
                        t1_valid_loss = time.time()
                        valid_loss_time = t1_valid_loss - t0_valid_loss
                    t0_gradient_step = time.time()
                    self.train_step_y_0(n, dataset)
                    t1_gradient_step = time.time()
                    gradient_step_time = t1_gradient_step - t0_gradient_step
                else:
                    if (iteration % self.net_config.logging_frequency) == 0:
                        try:
                            current_lr = self.optimizer_y_n.lr
                        except:
                            current_lr = self.optimizer_y_n._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                        # do not waste CPU time when it's not reported anyway
                        t0_valid_loss = time.time()
                        if self.is_validation_batching:
                            # since there is no automatic differentiation, this does not have to be done for time step 0
                            valid_loss = self.validation_batching(self.loss_y_n, n, valid_input_y_n, False).numpy()
                        else:
                            valid_loss = self.loss_y_n(n, valid_input_y_n, False).numpy()
                        t1_valid_loss = time.time()
                        valid_loss_time = t1_valid_loss - t0_valid_loss
                    t0_gradient_step = time.time()
                    self.train_step_y_n(n, dataset)
                    t1_gradient_step = time.time()
                    gradient_step_time = t1_gradient_step - t0_gradient_step
                # # # report
                if (iteration % self.net_config.logging_frequency) == 0:
                    iteration_time = time.time() - t0_iteration
                    step_time = time.time() - iteration_start
                    elapsed_time = time.time() - start_time
                    loss['Y'].append(valid_loss)
                    # format of step history was: header='step, learning_rate, loss_function, sampling_time, target_gathering_time, valid_loss_time, gradient_step_time, iteration_time, step_time, cumulative_elapsed_time',
                    to_append = [iteration, current_lr, valid_loss, sampling_time, target_gathering_time,
                                              valid_loss_time, gradient_step_time, iteration_time,
                                              step_time, elapsed_time]
                    step_history['Y'].append(to_append)
                    if self.net_config.verbose:
                        logging.info(
                            "iteration: %5u, lr: %.2e, val_loss: %.4e, iter_time: %.2e, step_time: %.2e, elapsed time: %3u"
                            % (iteration, current_lr, loss['Y'][-1], iteration_time, step_time, elapsed_time))
                t1_iteration = time.time()
            iteration_end = time.time()
            dummy_input_y = batch[0][0: 2, :]  # first element in tuple should always be x_n
            if n > 0:
                self.Y[n], self.Z[n], self.G[n] = self.model_snapshot_y_n(dummy_input_y)
            else:
                if not self.is_nn_at_t0:
                    self.Y[0] = self.model_y_0.Y0
                    self.Z[0] = self.model_y_0.Z0
                else:
                    self.Y[0], self.Z[0], self.G[0] = self.model_snapshot_y_0(dummy_input_y)



            logging.info("@avg epoch time: %.2e"
                         % ((iteration_end - iteration_start) / (self.net_config.num_iterations + 1)))
            logging.info('''::::::::::::::::::::::::::\nCompleted Y_%5u\n::::::::::::::::::::::::::''' % n)
            # # #
            training_history[n] = step_history  # list of dictionaries!
            self.generate_time_step_report(output_dir, n, training_history)

            if self.net_config.is_batch_norm:
                if n > 0:
                    self._switch_off_bn_after_last()
                    self._switch_off_bn_model_0()
                    logging.info("@BN: batch normalization parameters frozen for n<N-1!")

            if self.is_nn_at_t0:
                if n == 1:
                    self._transfer_parameters_to_t0_y0(dummy_input_y)
                    logging.info("@is_nn_at_t0==True: parameter transfer from models at n=1")

        self.is_trained = True
        if self.net_config.is_plot:
            self.plotter_pathwise(output_dir)
        logging.info("@pathwise estimates saved to file, quitting training now")

        return training_history

    def model_snapshot_y_n(self, dummy_input):
        Yn = SubNet(self.model_y_n.Yn.config, self.model_y_n.Yn.outputshape)
        Yn(dummy_input, True)
        if self.model_y_n.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_n.Yn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Yn.bn_layers[idx].trainable = False
        Yn.set_weights(self.model_y_n.Yn.get_weights())

        if not self.is_z_autodiff:
            Zn = SubNet(self.model_y_n.Zn.config, self.model_y_n.Zn.outputshape)
            Zn(dummy_input, True)
            if self.model_y_n.Zn.is_batch_norm:
                for idx, layer in enumerate(self.model_y_n.Zn.bn_layers):
                    if not layer.trainable:
                        # disable corresponding layer of Z as well
                        Zn.bn_layers[idx].trainable = False
            Zn.set_weights(self.model_y_n.Zn.get_weights())

            if hasattr(self.model_y_n, "Gn"):
                raise NotImplementedError
                Gn = SubNet(self.config, self.bsde.dim ** 2, is_mtx_output=True)
                Gn(dummy_input, True)

                Gn.set_weights(self.model_y_n.Gn.get_weights())
            else:
                Gn = None

        else:
            Zn = Yn.grad_call
            Gn = None



        return Yn, Zn, Gn

    def model_snapshot_y_0(self, dummy_input):
        Yn = SubNet(self.model_y_0.Yn.config, self.model_y_0.Yn.outputshape)
        Yn(dummy_input, True)
        if self.model_y_0.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Yn.bn_layers[idx].trainable = False
        Yn.set_weights(self.model_y_0.Yn.get_weights())

        if not self.is_z_autodiff:
            Zn = SubNet(self.model_y_0.Zn.config, self.model_y_0.Zn.outputshape)
            Zn(dummy_input, True)
            if self.model_y_0.Zn.is_batch_norm:
                for idx, layer in enumerate(self.model_y_0.Zn.bn_layers):
                    if not layer.trainable:
                        # disable corresponding layer of Z as well
                        Zn.bn_layers[idx].trainable = False
            Zn.set_weights(self.model_y_0.Zn.get_weights())

            if hasattr(self.model_y_0, "Gn"):
                Gn = SubNet(self.config, self.bsde.dim ** 2, is_mtx_output=True)
                Gn(dummy_input, True)
                Gn.set_weights(self.model_y_0.Gn.get_weights())
            else:
                Gn = None
        else:
            Zn = Yn.grad_call
            Gn = None
        return Yn, Zn, Gn

    def _switch_off_bn_after_last(self):
        if self.model_y_n.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_n.Yn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Y: frozen")
        if self.is_z_autodiff:
            raise NotImplementedError
        if self.model_y_n.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_n.Zn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Z: frozen")
        return 0

    def _switch_off_bn_model_0(self):
        if self.model_y_0.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Y: frozen")
        if self.is_z_autodiff:
            raise NotImplementedError
        if self.model_y_0.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_0.Zn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Z: frozen")

        return 0

    def _transfer_parameters_to_t0_y0(self, dummy_input):
        if self.bsde.N > 1:

            self.model_y_0.Yn(dummy_input, True)
            self.model_y_0.Yn.set_weights(self.model_y_n.Yn.get_weights())
            logging.info("@is_nn_at_t0==True: nn estimates at t0: Y: Y model transferred")

            if self.model_y_n.Yn.is_batch_norm:
                for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                    layer.trainable = False
                logging.info("@_transfer_at_t0:batch_norm: batch_normalization: Y: layers set to non-trainable")
                # raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")

            if not self.is_z_autodiff:
                self.model_y_0.Zn(dummy_input, True)
                self.model_y_0.Zn.set_weights(self.model_y_0.Zn.get_weights())

                if hasattr(self.model_y_0, "Gn"):
                    self.model_y_0.Gn(dummy_input, True)
                    self.model_y_0.Gn.set_weights(self.model_y_n.Gn.get_weights())
                else:
                    Gn = None

            if self.model_y_n.Zn.is_batch_norm:
                for idx, layer in enumerate(self.model_y_0.Zn.bn_layers):
                    layer.trainable = False
                logging.info("@_transfer_at_t0:batch_norm: batch_normalization: Z: layers set to non-trainable")
                # raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")

        return 0

class TwoLayerBaseSolver(BackwardBSDESolverBaseClass):
    def __init__(self, config, bsde, optimizer_func=custom_adam_optimizer):
        super(TwoLayerBaseSolver, self).__init__(config, bsde)
        self.is_has_gamma = True

        # # # Initialize Models
        self.model_y_0 = None
        self.model_y_n = None
        self.model_z_0 = None
        self.model_z_n = None

        # # #
        if not hasattr(self.net_config, "seed"):
            self.net_config.seed = None
            logging.info("@won't be seeding observations")

        # # # Initialize Optimizers
        # learning rate floor
        if not hasattr(self.net_config, "lower_bound_learning_rate"):
            self.net_config.lower_bound_learning_rate = None
            logging.info("@!!! no lower bound in learning rate schedule !!!")

        total_iterations = self.net_config.num_iterations * self.no_of_batches
        if (self.net_config.train_size == 0) or (self.net_config.train_size is None):
            total_iterations = self.net_config.num_iterations
            logging.info("@learning_rate_schedule: assuming resampling training")
        self.optimizer_y_0 = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)
        self.optimizer_y_n = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)
        self.optimizer_z_0 = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)
        self.optimizer_z_n = tf.keras.optimizers.Adam(learning_rate=ChenSchedule(self.net_config.initial_learning_rate,
                                                                                 self.net_config.decay_rate,
                                                                                 self.net_config.lower_bound_learning_rate,
                                                                                 total_iterations),
                                                      epsilon=1e-8)

        self.is_collect_future = False
        self.train_collected_future = None  # will be needed for the DP approach, to avoid recalculating already processed
        self.valid_collected_future = None  # None will default to not using them at all
        # time points at the start of each iteration step
        # this is a dict with keys: {'Y': ndarray, 'Z': ndarray, 'grad_Y': ndarray, 'jac_Z': ndarray}

        self.is_validation_batching = False  # whether the validation loss should be obtained by batching over the
        # validation dataset: this is needed to avoid constructing gigantic Jacobians for implicit schemes!

        self.is_z_autodiff = False
        self.is_gamma_autodiff = True

        if not hasattr(self.net_config, "pretrained_num_iterations"):
            self.net_config.pretrained_num_iterations = self.net_config.num_iterations
            logging.info("@no early termination for inner time points")

        if not hasattr(self.net_config, "is_t0_explicit_init"):
            self.net_config.is_t0_explicit_init = True
            logging.info("@doing t0 initialization according to explicit schemes by default")

        if not hasattr(self.eqn_config.discretization_config, "theta_y"):
            self.theta_y = None
        else:
            self.theta_y = self.eqn_config.discretization_config.theta_y

        if not hasattr(self.eqn_config.discretization_config, "theta_z"):
            self.theta_z = None
        else:
            self.theta_z = self.eqn_config.discretization_config.theta_z

        if not hasattr(self.net_config, "is_terminal_regression_preinit"):
            self.net_config.is_terminal_regression_preinit = False

        if not hasattr(self.net_config, "batch_size_pretrained"):
            self.net_config.batch_size_pretrained = None
        else:
            logging.info("@init: using different batch sizes for pretrained time steps")
            if self.net_config.train_size is not None:
                bn = self.net_config.train_size / self.net_config.batch_size_pretrained
                self.no_of_batches_pretrained = np.int(np.modf(bn)[1]) if np.modf(bn)[0] == 0 else np.int(
                    np.modf(bn)[1] + 1)
    def train_fetch(self, output_dir):
        logging.info("@train: fetch independent mini-batches separately")
        start_time = time.time()
        training_history = [None] * len(self.t)  # time step many long

        # this will be sampled for each iteration separately
        # train_data = self.bsde.sample(self.net_config.train_size)
        # tf.data.Dataset.from_tensor_slices(tensor_slices)
        valid_data = self.bsde.sample(self.net_config.valid_size)
        dw_valid, x_valid = valid_data


        # # # start iterating over time steps
        # # N
        self.Y[-1] = lambda X_T, training: self.bsde.g_tf(self.bsde.T, X_T)  # to have the same call format as nets
        self.Z[-1] = lambda X_T, training: tf.einsum('MJm,Mmn->MJn',
                                                     self.bsde.gradx_g_tf(self.bsde.T, X_T),
                                                     self.bsde.sigma_process_tf(self.bsde.T,X_T)
                                                     )

        logging.info("TERMINAL Z FIXED")
        self.generate_time_step_report(output_dir, self.bsde.N, training_history)
        if self.is_collect_future:
            raise NotImplementedError

        for n in range(self.bsde.N - 1, -1, -1):
            logging.info('''============================================================\nStarting Time Step %5u''' % n)
            # tf.keras.backend.clear_session()
            # logging.info("@train: keras.backend.(): state reset")
            if self.is_collect_future:
                raise NotImplementedError

            step_history = {'Z': [], 'Y': []}
            loss = {'Z': [], 'Y': []}

            # # # Z part first
            logging.info('''::::::::::::::::::::::::::\nZ_%5u''' % n)
            # x_valid_n = x_valid[:, :, n]
            # if n == self.bsde.num_time_interval - 1:
            #     self.get_last_jac_z_through_autodiff(x_valid_n)
            if n < self.bsde.N - 1:
                self._reset_optimizer(self.optimizer_z_n)
                logging.info("@Zn optimizer reset to initial state.")
                if n < (self.bsde.N - 1):
                    # # this only needs to be done once.
                    # if self.net_config.pretrained_num_iterations < self.net_config.num_iterations:
                    #     logging.info("@interior time step's total number of iterations changed.")
                    #     if n == 0:
                    #         logging.info("@ from: %d" % self.optimizer_z_0.learning_rate.total_epochs)
                    #         self.optimizer_z_0.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                    #         logging.info("@ to: %d" % self.optimizer_z_0.learning_rate.total_epochs)
                    #     else:
                    #         logging.info("@ from: %d" % self.optimizer_z_n.learning_rate.total_epochs)
                    #         self.optimizer_z_n.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                    #         logging.info("@ to: %d" % self.optimizer_z_n.learning_rate.total_epochs)
                    if self.net_config.pretrained_initial_learning_rate is not None:
                        # if self.net_config.is_pretrain_dump_only_for_y:
                        #     logging.info("@pretrained initial learning rate unchanged for Z")
                        #     pass
                        if n == 0:
                            if self.bsde.N > 1:
                                self.optimizer_z_0.learning_rate = self.lr_schedule_pretrained
                        else:
                            self.optimizer_z_n.learning_rate = self.lr_schedule_pretrained

                        logging.info("@interior time steps learning rate schedule updated")

            # # # will be fetched for each iteration independently
            # valid_input_z_n = self.get_inputs_z(n, valid_data, self.valid_collected_future)
            valid_input_z_n = self.get_inputs_z(n, valid_data)
            if n == 0:
                if not self.is_nn_at_t0:
                    if self.net_config.is_t0_explicit_init:
                        self.initialize_z_0(self.bsde.sample(self.net_config.batch_size))  # do it on an independent sample
                        logging.info("@smart t0 average initialization")

            iteration_start = time.time()
            total_iterations = self.net_config.num_iterations

            if n < self.bsde.N - 1:
                total_iterations = self.net_config.pretrained_num_iterations

            for iteration in range(total_iterations + 1):
                t0_iteration = time.time()
                train_data = self.bsde.sample(self.net_config.batch_size)
                t1_sampling = time.time()

                sampling_time = t1_sampling - t0_iteration

                batch = self.get_inputs_z(n, train_data)
                t1_target = time.time()
                dataset = batch
                target_gathering_time = t1_target - t1_sampling

                if n == 0:
                    # current_lr = np.nan
                    # current_lr = self.optimizer_z_0._learning_rate
                    try:
                        current_lr = self.optimizer_z_0.lr.numpy()
                    except:
                        current_lr = self.optimizer_z_0.learning_rate(iteration)
                    if (iteration % self.net_config.logging_frequency) == 0:
                        # only waste CPU time on this, when it's reported anyway
                        t0_valid_loss = time.time()
                        valid_loss = self.loss_z_0(n, valid_input_z_n, False).numpy()
                        t1_valid_loss = time.time()
                    t0_step = time.time()
                    self.train_step_z_0(n, dataset)
                    t1_step = time.time()
                else:
                    # current_lr = np.nan
                    # current_lr = self.optimizer_z_n._learning_rate._decayed_lr
                    # current_lr = self.optimizer_z_n.learning_rate(iteration)
                    try:
                        current_lr = self.optimizer_z_n.lr.numpy()
                    except:
                        current_lr = self.optimizer_z_n.learning_rate(iteration)

                    if (iteration % self.net_config.logging_frequency) == 0:
                        # only waste CPU time on this when it's reported anyway
                        t0_valid_loss = time.time()
                        if self.is_validation_batching:
                            # since there is no automatic differentiation, this does not have to be done for time step 0
                            valid_loss = self.validation_batching(self.loss_z_n, n, valid_input_z_n, False).numpy()
                        else:
                            valid_loss = self.loss_z_n(n, valid_input_z_n, False).numpy()
                        t1_valid_loss = time.time()
                    t1_preproc = time.time()
                    t0_step = time.time()
                    self.train_step_z_n(n, dataset)
                    t1_step = time.time()
                valid_loss_time = t1_valid_loss - t0_valid_loss
                gradient_step_time = t1_step - t0_step
                # # # report
                if (iteration % self.net_config.logging_frequency) == 0:
                    # print(self.get_targets_z.get_concrete_function(n, train_data))
                    # print(self.model_z_n.Zn.jacobian_call.pretty_printed_concrete_signatures())
                    iteration_time = time.time() - t0_iteration
                    step_time = time.time() - iteration_start
                    elapsed_time = time.time() - start_time
                    loss['Z'].append(valid_loss)
                    # step history has the following format
                    # header = 'step, learning_rate, loss_function, sampling_time, target_gathering_time, valid_loss_time, gradient_step_time, iteration_time, step_time, cumulative_elapsed_time',
                    to_append = [iteration, current_lr, loss['Z'][-1], sampling_time, target_gathering_time,
                                 valid_loss_time, gradient_step_time, iteration_time,
                                 step_time, elapsed_time]
                    # print(to_append)
                    step_history['Z'].append(to_append)
                    t1_iteration = time.time()
                    if self.net_config.verbose:
                        logging.info(
                            "iteration: %5u, lr: %.2e, val_loss: %.4e, iter_time: %.2e, step_time: %.2e, elapsed time: %3u"
                            % (iteration, current_lr, loss['Z'][-1], iteration_time, step_time, elapsed_time))
                del train_data
            iteration_end = time.time()
            dummy_input_z = batch[0][0: 2, :]  # first element in tuple should always be x_n
            if n > 0:
                self.Z[n], self.G[n] = self.model_snapshot_z_n(dummy_input_z)
            else:
                if not self.is_nn_at_t0:
                    self.Z[0] = self.model_z_0.Z0
                    if hasattr(self.model_z_0, "G0"):
                        self.G[0] = self.model_z_0.G0
                else:
                    self.Z[0], self.G[0] = self.model_snapshot_z_0(dummy_input_z)  # t=0 also has NN estimations

            # self.loss_y_n(self.bsde.N - 1, self.get_inputs_y(self.bsde.N - 1, valid_data), False)
            logging.info("@avg epoch time: %.2e"
                         %((iteration_end - iteration_start) / (self.net_config.num_iterations + 1)))
            logging.info('''::::::::::::::::::::::::::\nCompleted Z_%5u\n::::::::::::::::::::::::::''' % n)
            # # #

            # # # Y part second
            logging.info('''::::::::::::::::::::::::::\nY_%5u''' % n)
            if n < self.bsde.N - 1:
                self._reset_optimizer(self.optimizer_y_n)
                logging.info("@Yn optimizer reset to initial state.")
                if n < (self.bsde.N - 1):
                    # # this only needs to be done once.
                    # if self.net_config.pretrained_num_iterations < self.net_config.num_iterations:
                    #     logging.info("@interior time step's total number of iterations changed.")
                    #     if n == 0:
                    #         logging.info("@ from: %d" % self.optimizer_y_0.learning_rate.total_epochs)
                    #         self.optimizer_y_0.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                    #         logging.info("@ to: %d" % self.optimizer_y_0.learning_rate.total_epochs)
                    #     else:
                    #         logging.info("@ from: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                    #         self.optimizer_y_n.learning_rate.total_epochs = self.net_config.pretrained_num_iterations * self.no_of_batches
                    #         logging.info("@ to: %d" % self.optimizer_y_n.learning_rate.total_epochs)
                    if self.net_config.pretrained_initial_learning_rate is not None:
                        if n == 0:
                            if self.bsde.N > 1:
                                self.optimizer_y_0.learning_rate = self.lr_schedule_pretrained
                        else:
                            self.optimizer_y_n.learning_rate = self.lr_schedule_pretrained
                        logging.info("@interior time steps learning rate schedule updated")


            valid_input_y_n = self.get_inputs_y(n, valid_data, self.valid_collected_future)
            if n == 0:
                if not self.is_nn_at_t0:
                    if self.net_config.is_t0_explicit_init:
                        self.initialize_y_0(self.bsde.sample(self.net_config.batch_size))  # notice that this is the training data
                        logging.info("@smart t0 average initialization")

            if n < self.bsde.N - 1:
                total_iterations = self.net_config.pretrained_num_iterations
            iteration_start = time.time()
            for iteration in range(total_iterations + 1):
                t0_iteration = time.time()
                # # # maximum number of epochs per training step
                train_data = self.bsde.sample(self.net_config.batch_size)
                t1_sampling = time.time()

                sampling_time = t1_sampling - t0_iteration

                batch = self.get_inputs_y(n, train_data, self.valid_collected_future)
                t1_target = time.time()
                dataset = batch
                target_gathering_time = t1_target - t1_sampling

                if n == 0:
                    if (iteration % self.net_config.logging_frequency) == 0:
                        # only waste CPU time on this, when it's reported anyway
                        # current_lr = self.optimizer_y_0._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                        # current_lr = self.optimizer_y_0._learning_rate.numpy()
                        # current_lr = np.nan
                        try:
                            current_lr = self.optimizer_y_0.lr.numpy()
                        except:
                            current_lr = self.optimizer_y_0.learning_rate(iteration)

                        t0_valid_loss = time.time()
                        valid_loss = self.loss_y_0(n, valid_input_y_n, False).numpy()
                        t1_valid_loss = time.time()
                    t0_step = time.time()
                    self.train_step_y_0(n, dataset)
                    t1_step = time.time()
                else:
                    if (iteration % self.net_config.logging_frequency) == 0:
                        # current_lr = self.optimizer_y_n._decayed_lr(self.net_config.dtype)  # decay schedule hardcoded
                        # current_lr = self.optimizer_y_n._learning_rate.numpy()
                        # current_lr = np.nan
                        try:
                            current_lr = self.optimizer_y_n.lr.numpy()
                        except:
                            current_lr = self.optimizer_y_n.learning_rate(iteration)
                        t0_valid_loss = time.time()
                        if self.is_validation_batching:
                            # since there is no automatic differentiation, this does not have to be done for time step 0
                            valid_loss = self.validation_batching(self.loss_y_n, n, valid_input_y_n, False).numpy()
                        else:
                            valid_loss = self.loss_y_n(n, valid_input_y_n, False).numpy()
                        t1_valid_loss = time.time()
                    t1_preproc = time.time()
                    t0_step = time.time()
                    self.train_step_y_n(n, dataset)
                    t1_step = time.time()
                # # # report
                if (iteration % self.net_config.logging_frequency) == 0:
                    # only waste CPU time on this when it's reported anyway
                    valid_loss_time = t1_valid_loss - t0_valid_loss
                    gradient_step_time = t1_step - t0_step
                    iteration_time = time.time() - t0_iteration
                    step_time = time.time() - iteration_start
                    elapsed_time = time.time() - start_time
                    loss['Y'].append(valid_loss)
                    to_append = [iteration, current_lr, loss['Y'][-1], sampling_time, target_gathering_time,
                                 valid_loss_time, gradient_step_time, iteration_time, step_time, elapsed_time]
                    step_history['Y'].append(to_append)
                    t1_iteration = time.time()
                    if self.net_config.verbose:
                        if (iteration % self.net_config.logging_frequency) == 0:
                            logging.info(
                                "iteration: %5u, lr: %.2e, val_loss: %.4e, iter_time: %.2e, step_time: %.2e, elapsed time: %3u"
                                % (iteration, current_lr, loss['Y'][-1], iteration_time, step_time, elapsed_time))
            iteration_end = time.time()
            dummy_input_y = batch[0][0: 2, :]  # first element in tuple should always be x_n
            if n > 0:
                self.Y[n] = self.model_snapshot_y_n(dummy_input_y)
            else:
                if not self.is_nn_at_t0:
                    self.Y[0] = self.model_y_0.Y0  # single parameter estimations (fixed initial condition)
                else:
                    self.Y[0] = self.model_snapshot_y_0(dummy_input_y)

            if self.net_config.is_plot:
                self.plot_Yn(n, valid_data, output_dir + '/plots')
            logging.info("@avg epoch time: %.2e"
                         % ((iteration_end - iteration_start) / (self.net_config.num_iterations + 1)))
            logging.info('''::::::::::::::::::::::::::\nCompleted Y_%5u\n::::::::::::::::::::::::::''' % n)
            # # #
            training_history[n] = step_history  # list of dictionaries!
            self.generate_time_step_report(output_dir, n, training_history)

            if self.net_config.is_batch_norm:
                if n > 0:
                    self._reset_batch_norm_layers_moving_averages()
                    self._switch_off_bn_after_last()
                    self._switch_off_bn_model_0()
                    logging.info("@BN: batch normalization parameters frozen for n<N-1!")


            if self.is_nn_at_t0:
                if n == 1:
                    self._transfer_parameters_to_t0_z0(dummy_input_z)
                    self._transfer_parameters_to_t0_y0(dummy_input_y)
                    logging.info("@is_nn_at_t0==True: parameter transfer from models at n=1")

                    self._switch_off_bn_model_0()
                    logging.info("@is_nn_at_t0==True: batch normalization parameters frozen for t=0")

        self.is_trained = True

        if self.net_config.is_plot:
            self.plotter_pathwise(output_dir)
        logging.info("@pathwise estimates saved to file, quitting training now")

        return training_history

    def _transfer_parameters_to_t0_y0(self, dummy_input):
        if self.bsde.N > 1:
            # if self.model_y_n.is_batch_norm:
            #     raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")
            self.model_y_0.Yn(dummy_input, True)
            self.model_y_0.Yn.set_weights(self.model_y_n.Yn.get_weights())
            logging.info("@is_nn_at_t0==True: nn estimates at t0: Y: Y model transferred")

            if self.model_y_0.Yn.is_batch_norm:
                for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                    layer.trainable = False
                logging.info("@_transfer_at_t0:batch_norm: batch_normalization: Y: layers set to non-trainable")
                # raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")

        return 0

    def _transfer_parameters_to_t0_z0(self, dummy_input):
        if self.bsde.N > 1:
            # if self.model_z_n.Zn.is_batch_norm:
            #     raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")
            self.model_z_0.Zn(dummy_input, True)
            self.model_z_0.Zn.set_weights(self.model_z_n.Zn.get_weights())
            logging.info("@is_nn_at_t0==True: nn estimates at t0: Z: Z model transferred")

            if hasattr(self.model_z_0, "Gn") and (not self.is_gamma_autodiff):
                self.model_z_0.Gn(dummy_input, True)
                self.model_z_0.Gn.set_weights(self.model_z_n.Gn.get_weights())

            if self.model_z_0.Zn.is_batch_norm:
                for idx, layer in enumerate(self.model_z_0.Zn.bn_layers):
                    layer.trainable = False
                logging.info("@_transfer_at_t0:batch_norm: batch_normalization: Z: layers set to non-trainable")
                # raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")

            if self.model_z_0.Gn.is_batch_norm:
                for idx, layer in enumerate(self.model_z_0.Gn.bn_layers):
                    layer.trainable = False
                logging.info("@_transfer_at_t0:batch_norm: batch_normalization: Gamma: layers set to non-trainable")
                # raise NotImplementedError("Batch normalization with is_nn_at_t0 yet to be implemented")

        return 0

    def _switch_off_bn_after_last(self):
        if self.model_y_n.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_n.Yn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Y: frozen")
        if self.model_z_n.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_z_n.Zn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Z: frozen")
        if not self.is_gamma_autodiff:
            for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Gamma: frozen")

        return 0

    def _switch_off_bn_model_0(self):
        if self.model_y_0.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Y: frozen")
        if self.model_z_0.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_z_0.Zn.bn_layers):
                layer.trainable = False
            logging.info("@_switch_off_bn_after_last: Z: frozen")
        if not self.is_gamma_autodiff:
            if self.model_z_0.Gn.is_batch_norm:
                for idx, layer in enumerate(self.model_z_0.Gn.bn_layers):
                    layer.trainable = False
                logging.info("@_switch_off_bn_after_last: Gamma: frozen")

        return 0

    def _allow_full_forward_pass_to_initialize_moving_averages(self, n, training_data):
        # logging.info("@_allow_full_forward_pass: not doing anything")
        # return None
        """
        at the end of the previous time step, the model's moving averages have been reset to 0(1) for the mean(variance)
        in the batch normalization; however, the pre-initialized layers' weights are all inherently linked to these avgs
        in order to ease learning, we allow for one full forward pass of the training data (without any gradient step)
        in training mode for the batch normalization. this allows us to initialize the moving averages "close" to the
        expected behaviour
        :return: None (modifies corresponding models on the fly)
        """

        # return None

        if self.net_config.is_batch_norm:
            dw, x = training_data
            x_n = x[:, :, n]

            if self.model_y_n.is_batch_norm:
                # [logging.info(layer.momentum) for layer in self.model_y_n.bn_layers]
                org_momentums = [layer.momentum for layer in self.model_y_n.bn_layers]
                for idx, layer in enumerate(self.model_y_n.bn_layers):
                    self.model_y_n.bn_layers[idx].momentum = 0
                self.model_y_n(x_n, True)  # zero won't be rolled
                # [logging.info(layer.momentum) for layer in self.model_y_n.bn_layers]
                for idx, layer in enumerate(self.model_y_n.bn_layers):
                    self.model_y_n.bn_layers[idx].momentum = org_momentums[idx]
                del org_momentums
                # [logging.info(layer.momentum) for layer in self.model_y_n.bn_layers]
                logging.info("@batch normalization: Z model: momentum temporarily changed for reinitalization")
                logging.info("@batch_normalization: Y model: completed full forward pass to initialize moving avgs")

            if self.model_z_n.Zn.is_batch_norm:
                org_momentums = [layer.momentum for layer in self.model_z_n.Zn.bn_layers]
                for idx, layer in enumerate(self.model_z_n.Zn.bn_layers):
                    self.model_z_n.Zn.bn_layers[idx].momentum = 0
                self.model_z_n.Zn(x_n, True)
                for idx, layer in enumerate(self.model_z_n.Zn.bn_layers):
                    self.model_z_n.Zn.bn_layers[idx].momentum = org_momentums[idx]
                del org_momentums
                logging.info("@batch normalization: Z model: Zn: momentum temporarily changed for reinitalization")
                logging.info("@batch_normalization: Z model: Zn: completed full forward pass to initialize moving avgs")
                if hasattr(self.model_z_n, "Gn") and (not self.is_gamma_autodiff):
                    org_momentums = [layer.momentum for layer in self.model_z_n.Gn.bn_layers]
                    for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
                        self.model_z_n.Gn.bn_layers[idx].momentum = 0
                    self.model_z_n.Gn(x_n, True)
                    for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
                        self.model_z_n.Gn.bn_layers[idx].momentum = org_momentums[idx]
                    del org_momentums
                    logging.info("@batch normalization: Z model: Gn: momentum temporarily changed for reinitalization")
                    logging.info("@batch_normalization: Z model: Zn: completed full forward pass to initialize moving avgs")
        return None

    def _reset_batch_norm_layers_moving_averages(self):
        """
        after a training step is completed, we transferred not just the weights of the model, but also the moving
        averages and variances of the batch normalization layers. however, those are inherently related to time step n+1
        and shall not be used at time step n

        this methods takes both models (self.model_y_n, self.model_z_n) and overwrites (reinitializes) the corresponding layers

        in order to this we follow the same procedure as in the initialization of the batchnormalization source code
        under: https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/normalization.py

        :return: None (objects modified on the fly)
        # """
        logging.info("@_reset_moving_averages: not doing anything")
        return None

        # return None

        if self.model_y_n.is_batch_norm:
            # Y model
            for idx, layer in enumerate(self.model_y_n.bn_layers):
                # print(idx)
                # print(layer.moving_mean)
                mean_shape = layer.moving_mean.shape
                mean_dtype = layer.moving_mean.dtype
                layer.moving_mean.assign(tf.zeros(mean_shape, dtype=mean_dtype))

                variance_shape = layer.moving_variance.shape
                variance_dtype = layer.moving_variance.dtype
                layer.moving_variance.assign(tf.ones(variance_shape, dtype=variance_dtype))

            logging.info("@batch_normalization: Y model: reinitialized moving averages")

        if self.model_z_n.Zn.is_batch_norm:
            # Z model: Zn
            for idx, layer in enumerate(self.model_z_n.Zn.bn_layers):
                mean_shape = layer.moving_mean.shape
                mean_dtype = layer.moving_mean.dtype
                layer.moving_mean.assign(tf.zeros(mean_shape, dtype=mean_dtype))

                variance_shape = layer.moving_variance.shape
                variance_dtype = layer.moving_variance.dtype
                layer.moving_variance.assign(tf.ones(variance_shape, dtype=variance_dtype))

            logging.info("@batch_normalization: Z model: Zn: reinitialized moving averages")

            # Z model: Gn
            if hasattr(self.model_z_n, "Gn") and (not self.is_gamma_autodiff):
                for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
                    mean_shape = layer.moving_mean.shape
                    mean_dtype = layer.moving_mean.dtype
                    layer.moving_mean.assign(tf.zeros(mean_shape, dtype=mean_dtype))

                    variance_shape = layer.moving_variance.shape
                    variance_dtype = layer.moving_variance.dtype
                    layer.moving_variance.assign(tf.ones(variance_shape, dtype=variance_dtype))

        return None

    def model_snapshot_z_n(self, dummy_input):
        Zn = SubNet(self.model_z_n.Zn.config, self.model_z_n.Zn.outputshape)
        Zn(dummy_input, True)
        if self.model_z_n.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_z_n.Zn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Zn.bn_layers[idx].trainable = False
        Zn.set_weights(self.model_z_n.Zn.get_weights())
        Zn.trainable = False
        logging.info("@model_snapshot_z_n: Z: trained model frozen")

        if hasattr(self.model_z_n, "Gn") and (not self.is_gamma_autodiff):
            # raise NotImplementedError
            Gn = SubNet(self.model_z_n.Gn.config, self.model_z_n.Gn.outputshape)
            Gn(dummy_input, True)
            if self.model_z_n.Gn.is_batch_norm:
                for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
                    if not layer.trainable:
                        # disable corresponding layer of Z as well
                        Gn.bn_layers[idx].trainable = False
            # for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
            #     Gn.bn_layers[idx].trainable = self.model_z_n.Gn.bn_layers[idx].trainable
            # Gn(dummy_input, self.net_config.is_batch_norm_last_only)
            # logging.info("@saving:Gn: com e back here, i am changing the save structure")
            Gn.set_weights(self.model_z_n.Gn.get_weights())
            Gn.trainable = False
            logging.info("@model_snapshot_z_n: Gamma: trained model frozen")
        else:
            Gn = None

        return Zn, Gn

    def model_snapshot_z_0(self, dummy_input):
        Zn = SubNet(self.model_z_0.Zn.config, self.model_z_0.Zn.outputshape)
        Zn(dummy_input, True)
        if self.model_z_0.Zn.is_batch_norm:
            for idx, layer in enumerate(self.model_z_0.Zn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Zn.bn_layers[idx].trainable = False
        Zn.set_weights(self.model_z_0.Zn.get_weights())
        Zn.trainable = False
        logging.info("@model_snapshot_z_n: Z: trained model frozen")

        if hasattr(self.model_z_0, "Gn") and (not self.is_gamma_autodiff):
            # raise NotImplementedError
            Gn = SubNet(self.model_z_0.Gn.config, self.model_z_0.Gn.outputshape)
            Gn(dummy_input, True)
            if self.model_z_0.Gn.is_batch_norm:
                for idx, layer in enumerate(self.model_z_0.Gn.bn_layers):
                    if not layer.trainable:
                        # disable corresponding layer of Z as well
                        Gn.bn_layers[idx].trainable = False
            # for idx, layer in enumerate(self.model_z_n.Gn.bn_layers):
            #     Gn.bn_layers[idx].trainable = self.model_z_n.Gn.bn_layers[idx].trainable
            # Gn(dummy_input, self.net_config.is_batch_norm_last_only)
            # logging.info("@saving:Gn: com e back here, i am changing the save structure")
            Gn.set_weights(self.model_z_0.Gn.get_weights())
            Gn.trainable = False
            logging.info("@model_snapshot_z_0: Gamma: trained model frozen")
        else:
            Gn = None

        return Zn, Gn

    def model_snapshot_y_n(self, dummy_input):
        Yn = SubNet(self.model_y_n.Yn.config, self.model_y_n.Yn.outputshape)
        Yn(dummy_input, True)
        if self.model_y_n.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_n.Yn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Yn.bn_layers[idx].trainable = False
        Yn.set_weights(self.model_y_n.Yn.get_weights())
        Yn.trainable = False

        logging.info("@model_snapshot_y_n: trained model frozen")

        return Yn

    def model_snapshot_y_0(self, dummy_input):
        Yn = SubNet(self.model_y_0.Yn.config, self.model_y_0.Yn.outputshape)
        Yn(dummy_input, True)
        if self.model_y_0.Yn.is_batch_norm:
            for idx, layer in enumerate(self.model_y_0.Yn.bn_layers):
                if not layer.trainable:
                    # disable corresponding layer of Z as well
                    Yn.bn_layers[idx].trainable = False
        Yn.set_weights(self.model_y_0.Yn.get_weights())
        Yn.trainable = False

        logging.info("@model_snapshot_y_0: trained model frozen")

        return Yn
