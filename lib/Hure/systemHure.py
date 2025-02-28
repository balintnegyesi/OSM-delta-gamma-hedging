import tensorflow as tf
import numpy as np
import logging
from lib.base_solvers.models_backward_portfolio import BaseBackwardModel
# from SeparatedDiscreteSolverThetaHybrids import SeparatedDiscreteSolverThetaHybrids
from lib.base_solvers.solvers_backward import OneLayerBaseSolver as SeparatedDiscreteSolverThetaHybrids
from lib.SubNet import SubNet
from lib.base_solvers.models_backward_portfolio import HureModeln
from lib.misc.learning_rate_schedules import ChenSchedule


class systemHure(SeparatedDiscreteSolverThetaHybrids):
    """
        this inheritence is only here so that I don't have to rewrite all convenience functions (plotting, saving, etc.)
        training and co. will be overwritten though
    """
    def __init__(self, config, bsde, is_z_autodiff=False):
        super(systemHure, self).__init__(config, bsde)

        # # # Initialize Models
        if not self.is_nn_at_t0:
            raise NotImplementedError
        else:
            self.model_y_0 = HureModeln(config, bsde, self.is_z_autodiff)
        self.model_y_n = HureModeln(config, bsde, self.is_z_autodiff)
        self.model_z_0 = None
        self.model_z_n = None

        # # # implicit models may have gamma estimates for t = 0
        self.G0 = None

        # # # Initialize Optimizers
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.net_config.initial_learning_rate,
                                                                     self.net_config.num_iterations,
                                                                     self.net_config.decay_rate)
        self.lr_schedule_pretrained = tf.keras.optimizers.schedules.ExponentialDecay(
            self.net_config.pretrained_initial_learning_rate,
            self.net_config.pretrained_num_iterations,
            self.net_config.initial_learning_rate / self.net_config.pretrained_initial_learning_rate
            * self.net_config.decay_rate)

        if not hasattr(self.net_config, "is_chen_schedule"):
            self.net_config.is_chen_schedule = False
        if self.net_config.is_chen_schedule:
            lr_schedule = ChenSchedule(self.net_config.initial_learning_rate,
                                       self.net_config.decay_rate, self.net_config.lower_bound_learning_rate,
                                       self.net_config.num_iterations)
            self.lr_schedule_pretrained = ChenSchedule(self.net_config.pretrained_initial_learning_rate,
                                                       self.net_config.decay_rate,
                                                       self.net_config.lower_bound_learning_rate,
                                                       self.net_config.pretrained_num_iterations)

        self.optimizer_y_n = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        self.optimizer_y_0 = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        self.optimizer_z_0 = None
        self.optimizer_z_n = None


        self.is_collect_future = False
        self.train_collected_future = None  # will be needed for the DP approach, to avoid recalculating already processed
        self.valid_collected_future = None  # None will default to not using them at all
        # time points at the start of each iteration step
        # this is a dict with keys: {'Y': ndarray, 'Z': ndarray, 'grad_Y': ndarray, 'jac_Z': ndarray}

        self.is_validation_batching = False  # whether the validation loss should be obtained by batching over the
        # validation dataset: this is needed to avoid constructing gigantic Jacobians for implicit schemes!

        if not hasattr(self.eqn_config.discretization_config, "theta_y"):
            self.theta_y = 1
        else:
            self.theta_y = self.eqn_config.discretization_config.theta_y
            self.theta_y = 1
            logging.info("@CPME BACK HERE: I FIXED THETA_Y FOR THE BACKWARD DEEP BSDE METHOD")

        logging.info("@@@ Theta: %.2e"%self.theta_y)

    # # # Yn
    @tf.function
    def loss_y_n(self, n, inputs, training):
        dw_n, x_n, target_n = inputs

        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        c_n = self.model_y_n.Yn(x_n, training)
        z_n = self.model_y_n.Zn(x_n, training)
        if self.is_z_autodiff:
            raise NotImplementedError

        est = c_n - self.theta_y * delta_t * self.bsde.f_tf(t_n, x_n, c_n, z_n) + \
              tf.einsum("Mjm, Mm -> Mj", z_n, dw_n)
        delta = est - target_n


        loss = tf.reduce_mean(tf.linalg.norm(delta, axis=-1, ord=2) ** 2)  # L^2

        return loss

    # # # Y0
    @tf.function
    def loss_y_0(self, n, inputs, training):
        dw_n, x_n, target_n = inputs

        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        if not self.is_nn_at_t0:
            raise NotImplementedError
        else:
            c_n = self.model_y_0.Yn(x_n, training)
            z_n = self.model_y_0.Zn(x_n, training)

        if self.is_z_autodiff:
            raise NotImplementedError

        est = c_n - self.theta_y * delta_t * self.bsde.f_tf(t_n, x_n, c_n, z_n)+ \
              tf.einsum("Mjm, Mm -> Mj", z_n, dw_n)
        delta = est - target_n

        loss = tf.reduce_mean(tf.linalg.norm(delta, axis=-1, ord=2) ** 2)

        return loss

    @tf.function
    def get_targets_y(self, n, data):
        dw, x = data
        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n

        x_next = x[:, :, n + 1]
        c_next = self.Y[n + 1](x_next, False)
        z_next = self.Z[n + 1](x_next, False)

        l_next = self.bsde.g_tf(t_next, x_next)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_next > c_next, 1.0, 0.0), self.bsde.is_exercise_date[..., n])
        y_next = c_next + is_exercised * (l_next - c_next)

        target_n = y_next + (1 - self.theta_y) * delta_t * self.bsde.f_tf(t_next, x_next, c_next, z_next)

        return target_n
    @tf.function
    def get_inputs_y(self, n, data, collected_future=None):
        dw, x = data
        dw_n = dw[:, :, n]
        x_n = x[:, :, n]



        targets_n = self.get_targets_y(n, data)

        return dw_n, x_n, targets_n

    def initialize_y_0_z_0(self, train_input):
        raise NotImplementedError
        dw, x = train_input
        t_1 = self.bsde.t[1]
        x_1 = x[:, :, 1]
        dw_0 = dw[:, :, 0]
        y_1 = self.Y[1](x_1, False)
        if (self.is_z_autodiff) and (0 < self.bsde.N - 1):
            grad_y_1 = self.Y[1].grad_call(x_1, False).numpy()
            sigma_1 = self.bsde.ForwardDiffusion.sigma_process_tf(t_1, x_1).numpy()
            z_1 = np.einsum('aij, aj -> ai', sigma_1, grad_y_1)
        else:
            z_1 = self.Z[1](x_1, False).numpy()
        projection_y = y_1 + self.bsde.delta_t * self.bsde.f_tf(t_1, x_1, y_1, z_1)
        projection_z = y_1 * dw_0 / self.bsde.delta_t

        self.model_y_0.Y0.assign(tf.reshape(tf.reduce_mean(projection_y), shape=[1]))
        self.model_y_0.Z0.assign(tf.reshape(tf.reduce_mean(projection_z, axis=0), shape=[1, self.bsde.dim]))


        return None