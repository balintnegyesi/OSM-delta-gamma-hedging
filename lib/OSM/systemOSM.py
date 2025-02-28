import tensorflow as tf
import numpy as np
from lib.base_solvers.solvers_backward import TwoLayerBaseSolver
from lib.base_solvers.models_backward_portfolio import OSMModelYn, OSMModelZn
from lib.misc.learning_rate_schedules import ChenSchedule
import logging
import copy

class systemOSM(TwoLayerBaseSolver):
    """
        this inheritence is only here so that I don't have to rewrite all convenience functions (plotting, saving, etc.)
        training and co. will be overwritten though
    """
    def __init__(self, config, bsde):
        super(systemOSM, self).__init__(config, bsde)

        self.model_y_n = OSMModelYn(self.config, self.bsde)
        self.model_y_0 = OSMModelYn(self.config, self.bsde)

        self.model_z_n = OSMModelZn(self.config, self.bsde)
        self.model_z_0 = OSMModelZn(self.config, self.bsde)

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

        self.optimizer_z_n = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        self.optimizer_y_n = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        self.optimizer_z_0 = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)
        self.optimizer_y_0 = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-8)

        self.is_gamma_autodiff = False

        logging.info("@Disc: theta_y=%.2e" % self.theta_y)
        logging.info("@Disc: theta_z=%.2e" % self.theta_z)
        print("Is Yn batch_normalized: ", self.model_y_n.Yn.is_batch_norm)
        print("Is Zn batch_normalized: ", self.model_z_n.Zn.is_batch_norm)

    # # # Zn: neural network parametrization
    @tf.function(reduce_retracing=True)  # decorated so that calculation of validation losses is less expensive
    def loss_z_n(self, n, inputs, training):
        dw_n, x_n, nabla_z_f_next, targets_n = inputs
        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n

        # initial condition for Malliavin derivative starting off from x_n at t_n
        sigma_n = self.bsde.sigma_process_tf(t_n, x_n)
        DnXn = sigma_n  # DnXk: M x m x m
        z_n = self.model_z_n.Zn(x_n, training)  # M x j x m
        gamma_n = self.model_z_n.Gn(x_n, training)  # M x j x m x m

        DnZn = tf.einsum("Mjmn, Mnl -> Mjml", gamma_n, DnXn)

        time_part_driver_z = tf.einsum("Mjqn, Mqnm -> Mjm", nabla_z_f_next, DnZn)

        ito_part = tf.einsum("Mjml, Mm -> Mjl", DnZn, dw_n)

        est_z = z_n - delta_t * time_part_driver_z + ito_part  # time part fully explicit
        delta_z = est_z - targets_n  # M x J x d


        delta_norm = tf.norm(delta_z, axis=-1, ord=2) ** 2  # M x J -- for each option
        loss = tf.reduce_mean(tf.reduce_sum(delta_norm, axis=-1))


        # gather regularization terms if defined within the model
        if self.model_z_n.Zn.is_l2_regularization:
            reg_losses = self.model_z_n.Zn.losses
            l2_regularization_loss = 0
            for w in reg_losses:
                l2_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_regularization_loss
        if self.model_z_n.Gn.is_l2_regularization:
            reg_losses = self.model_z_n.Gn.losses
            l2_Gn_regularization_loss = 0
            for w in reg_losses:
                l2_Gn_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_Gn_regularization_loss

        return loss

    # # # Z0: in case of single parameter training at t=0
    @tf.function(reduce_retracing=True)
    def loss_z_0(self, n, inputs, training):
        dw_n, x_n, nabla_z_f_next, targets_n = inputs
        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n

        # initial condition for Malliavin derivative starting off from x_n at t_n
        sigma_n = self.bsde.sigma_process_tf(t_n, x_n)
        DnXn = sigma_n  # DnXk: M x m x m
        z_n = self.model_z_0.Zn(x_n, training)  # M x j x m
        gamma_n = self.model_z_0.Gn(x_n, training)  # M x j x m x m

        DnZn = tf.einsum("Mjmn, Mnl -> Mjml", gamma_n, DnXn)

        time_part_driver_z = tf.einsum("Mjqn, Mqnm -> Mjm", nabla_z_f_next, DnZn)

        ito_part = tf.einsum("Mjml, Mm -> Mjl", DnZn, dw_n)

        est_z = z_n - delta_t * time_part_driver_z + ito_part  # time part fully explicit
        delta_z = est_z - targets_n  # M x J x d


        delta_norm = tf.norm(delta_z, axis=-1, ord=2) ** 2  # M x J -- for each option
        # loss = tf.reduce_mean(tf.reduce_max(delta_norm, axis=-1))
        loss = tf.reduce_mean(tf.reduce_sum(delta_norm, axis=-1))

        # gather regularization terms if defined within the model
        if self.model_z_0.Zn.is_l2_regularization:
            reg_losses = self.model_z_0.Zn.losses
            l2_regularization_loss = 0
            for w in reg_losses:
                l2_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_regularization_loss

        if self.model_z_0.Gn.is_l2_regularization:
            reg_losses = self.model_z_0.Gn.losses
            l2_Gn_regularization_loss = 0
            for w in reg_losses:
                l2_Gn_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_Gn_regularization_loss

        return loss
    @tf.function(reduce_retracing=True)
    def loss_y_n(self, n, inputs, training):
        dw_n, x_n, z_tilde_n, targets_n = inputs

        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        c_n = self.model_y_n.Yn(x_n, training)


        payoff = self.bsde.g_tf(t_n, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(payoff > c_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n])

        sigma_n = self.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", self.bsde.gradx_g_tf(t_n, x_n), sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        est = est = c_n - self.theta_y * delta_t * self.bsde.f_tf(t_n, x_n, c_n, z_n) + \
                    tf.einsum("Mjm, Mm -> Mj", z_n, dw_n)
        delta = est - targets_n
        loss = tf.reduce_mean(tf.linalg.norm(delta, axis=-1, ord=2) ** 2)  # L^2

        # gather regularization terms if defined within the model
        if self.model_y_n.Yn.is_l2_regularization:
            reg_losses = self.model_y_n.Yn.losses
            l2_regularization_loss = 0
            for w in reg_losses:
                l2_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_regularization_loss

        return loss
    # # # Y0: in case of single parameter training at t=0
    @tf.function(reduce_retracing=True)
    def loss_y_0(self, n, inputs, training):
        dw_n, x_n, z_tilde_n, targets_n = inputs

        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        c_n = self.model_y_0.Yn(x_n, training)

        payoff = self.bsde.g_tf(t_n, x_n)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(payoff > c_n, 1.0, 0.0), self.bsde.is_exercise_date[..., n])

        sigma_n = self.bsde.sigma_process_tf(t_n, x_n)
        z_exercised_n = tf.einsum("MJd, Mdq -> MJq", self.bsde.gradx_g_tf(t_n, x_n), sigma_n)
        z_n = z_tilde_n + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_n - z_tilde_n)

        est = est = c_n - self.theta_y * delta_t * self.bsde.f_tf(t_n, x_n, c_n, z_n) + \
                    tf.einsum("Mjm, Mm -> Mj", z_n, dw_n)
        delta = est - targets_n
        loss = tf.reduce_mean(tf.linalg.norm(delta, axis=-1, ord=2) ** 2)  # L^2

        # gather regularization terms if defined within the model
        if self.model_y_0.Yn.is_l2_regularization:
            reg_losses = self.model_y_0.Yn.losses
            l2_regularization_loss = 0
            for w in reg_losses:
                l2_regularization_loss += tf.reduce_sum(w)  # sum of all regularization terms corresponding to
                # parameters of a given layer
            loss += l2_regularization_loss

        return loss


    @tf.function(reduce_retracing=True)
    def get_targets_z(self, n, data):
        # print("get_targets_z: retracing")
        dw, x = data
        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        x_next = x[:, :, n + 1]
        c_next = self.Y[n + 1](x_next, False)  # this is the continuation value
        z_tilde_next = self.Z[n + 1](x_next, False)

        # Malliavin arguments
        DnXnext = self.bsde.sample_step_malliavin(dw, x, n)  # gets D_nX_{n+1} from X_n with a forward
        inv_sigma_next = self.bsde.inverse_sigma_process_tf(t_next, x_next)
        grad_c_next = tf.einsum('Mjm, Mmn -> Mjn', z_tilde_next, inv_sigma_next)
        DnCnext = tf.einsum('Mjn, Mnm -> Mjm', grad_c_next, DnXnext)  # Malliavin chain rule
        DnLnext = tf.einsum('MJm, Mmn -> MJn', self.bsde.gradx_g_tf(t_next, x_next), DnXnext)

        l_next = self.bsde.g_tf(t_next, x_next)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_next > c_next, 1.0, 0.0), self.bsde.is_exercise_date[..., n])

        grad_payoff_next = self.bsde.gradx_g_tf(t_next, x_next)
        sigma_next = self.bsde.sigma_process_tf(t_next, x_next)
        exercised_z_next = tf.einsum("MJn, Mnm -> MJm", grad_payoff_next, sigma_next)

        z_next = z_tilde_next + tf.einsum("MJ, MJd -> MJd", is_exercised, exercised_z_next - z_tilde_next)
        reflection_target = DnCnext + tf.einsum("MJ, MJd -> MJd", is_exercised, DnLnext - DnCnext)


        gradx_f_next = self.bsde.gradx_f_tf(t_next, x_next, c_next, z_next)
        grady_f_next = self.bsde.grady_f_tf(t_next, x_next, c_next, z_next)
        gradz_f_next = self.bsde.gradz_f_tf(t_next, x_next, c_next, z_next)

        driver_x = tf.einsum('Mjm, Mmn -> Mjn', gradx_f_next, DnXnext)
        driver_y = tf.einsum('MjJ, MJm -> Mjm', grady_f_next, DnCnext)
        f_d_driver_x_driver_y = driver_x + driver_y

        target_z_n = reflection_target + delta_t * f_d_driver_x_driver_y  # fully explicit in time

        return gradz_f_next, target_z_n
    @tf.function(reduce_retracing=True)
    def get_inputs_z(self, n, data):
        # note: collected future defaults to none since self.is_collect_future defaults to none
        # print("get_inputs_z: retracing")
        dw, x = data
        dw_n = dw[:, :, n]
        x_n = x[:, :, n]
        nabla_z_f_next, targets_n = self.get_targets_z(n, data)

        return dw_n, x_n, nabla_z_f_next, targets_n
    @tf.function(reduce_retracing=True)
    def get_targets_y(self, n, data):
        dw, x = data
        t_n = self.bsde.t[n]
        t_next = self.bsde.t[n + 1]
        delta_t = t_next - t_n
        x_next = x[:, :, n + 1]
        c_next = self.Y[n + 1](x_next, False)
        z_tilde_next = self.Z[n + 1](x_next, False)

        l_next = self.bsde.g_tf(t_next, x_next)
        is_exercised = tf.einsum("MJ, J -> MJ", tf.where(l_next > c_next, 1.0, 0.0), self.bsde.is_exercise_date[..., n])
        y_next = c_next + is_exercised * (l_next - c_next)

        grad_payoff_next = self.bsde.gradx_g_tf(t_next, x_next)
        sigma_next = self.bsde.sigma_process_tf(t_next, x_next)
        z_exercised_next = tf.einsum("Mjm, Mmn -> Mjn", grad_payoff_next, sigma_next)
        z_next = z_tilde_next + tf.einsum("MJ, MJd -> MJd", is_exercised, z_exercised_next - z_tilde_next)

        f_next = self.bsde.f_tf(t_next, x_next, c_next, z_next)
        target_n = y_next + (1 - self.theta_y) * delta_t * f_next

        return target_n

    @tf.function(reduce_retracing=True)
    def get_inputs_y(self, n, data, collected_future=None):
        # note: collected future defaults to none since self.is_collect_future defaults to none
        dw, x = data
        dw_n = dw[:, :, n]
        x_n = x[:, :, n]

        z_n = self.Z[n](x_n, False)
        targets_n = self.get_targets_y(n, data)

        return dw_n, x_n, z_n, targets_n