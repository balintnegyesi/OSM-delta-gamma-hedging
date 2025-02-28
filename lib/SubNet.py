import tensorflow as tf
import numpy as np
import warnings
import logging

def requ(x):
    """
    rectified quadratic unit activation: max(0, x^2)
    :param x:
    :return:
    """
    return tf.where(x > 0, tf.square(x), 0)

def bent_identity(x):
    """
    bent identity, C^\infty, monotonic derivative, approximates identity around 0: [sqrt(x^2+1)-1]/2+x
    :param x:
    :return:
    """
    return (tf.sqrt(tf.square(x) + 1) - 1) / 2 + x

class SubNet(tf.keras.Model):
    def __init__(self, config, output_dim, is_mtx_output=False):
        # output dim is needed in addition to be able to use the same class for networks estimating Y and Z
        super(SubNet, self).__init__()
        self.config = config
        self.output_dim = output_dim

        self.output_dim = np.prod(output_dim).astype("int")
        if np.isscalar(output_dim):
            self.outputshape = [self.output_dim]
        else:
            self.outputshape = output_dim

        self.is_mtx_output = is_mtx_output

        self.is_bias = config.net_config.is_bias
        if not hasattr(config.net_config, "is_batch_norm"):
            self.is_batch_norm = False
            logging.info("@SubNet:init: no batch normalization by default")
        else:
            self.is_batch_norm = config.net_config.is_batch_norm
        logging.info("@SubNet:init: is_batch_norm=%s" %str(self.is_batch_norm))

        if not hasattr(config.net_config, "is_layer_norm"):
            self.is_layer_norm = True
            logging.info("@SubNet:init: layer normalization by default")
        else:
            self.is_layer_norm = config.net_config.is_layer_norm
        logging.info("@SubNet:init: is_layer_norm=%s" %str(self.is_layer_norm))

        num_hiddens = config.net_config.num_hiddens

        if not hasattr(config.net_config, "activation_func"):
            self.activation_func = 'relu'
            warnings.warn('No Activation Function Provided: will be using default ReLU')
        else:
            self.activation_func = config.net_config.activation_func

        if not hasattr(config.net_config, "bn_epsilon"):
            self.bn_epsilon = 1e-3  # default
        else:
            self.bn_epsilon = config.net_config.bn_epsilon
        logging.info('@SubNet:init: batch normalization: using epsilon=%.2e' %self.bn_epsilon)
        self.bn_epsilon = tf.cast(self.bn_epsilon, dtype=tf.keras.backend.floatx()).numpy()

        if self.is_batch_norm:
            self.bn_layers = [tf.keras.layers.BatchNormalization(epsilon=self.bn_epsilon) for _ in range(len(num_hiddens) + 1)]
        else:
            self.bn_layers = None

        if self.is_layer_norm:
            if self.is_batch_norm:
                raise ValueError("@SubNet:init: batch and layer normalization together forbidden")
            self.layer_layers = [tf.keras.layers.LayerNormalization(epsilon=self.bn_epsilon) for _ in range(len(num_hiddens))]

        if not hasattr(config.net_config, "is_l2_regularization"):
            self.is_l2_regularization = False
            logging.info("@SubNet:init: L2 regularization defaults to False")
        else:
            self.is_l2_regularization = config.net_config.is_l2_regularization
            l2_regularization_scale = config.net_config.l2_regularization_scale
        logging.info("@SubNet:init: is_l2_regulariation=%s" %str(self.is_l2_regularization))
        if self.is_l2_regularization:
            logging.info("@SubNet:init: l2_regularization scale=%.2e" %l2_regularization_scale)
            self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                       use_bias=self.is_bias,
                                                       activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_scale))
                                 for i in range(len(num_hiddens))]  # tfa.layers.WeightNormalization
        else:
            self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=self.is_bias,
                                                   activation=None)
                             for i in range(len(num_hiddens))]

        if not hasattr(config.net_config, "is_dropout"):
            self.is_dropout = False
            logging.info("@SubNet:init: is_dropout defaults to False")
        else:
            self.is_dropout = config.net_config.is_dropout
        logging.info("@SubNet:init: is_dropout=%s" %str(self.is_dropout))
        if self.is_dropout:
            logging.info("@SubNet: DROPOUT")
            self.dropout_rate = config.net_config.dropout_rate
            self.dropout_layers = [tf.keras.layers.AlphaDropout(self.dropout_rate) for _ in range(len(num_hiddens))]

        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(self.output_dim, use_bias=True, activation=None))
        self.reshape_layer = tf.keras.layers.Reshape(self.outputshape)
        # if self.is_mtx_output:
        #     d = int(np.sqrt(output_dim))  # output dim is the flattened vector
        #     self.reshape_layer = tf.keras.layers.Reshape((d, d))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        if self.is_batch_norm:
            x = self.bn_layers[0](x, training)

        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)

            if self.activation_func == 'requ':
                x = requ(x)
            elif self.activation_func == "bent_identity":
                x = bent_identity(x)
            else:
                x = tf.keras.layers.Activation(self.activation_func)(x)

            if self.is_batch_norm:
                x = self.bn_layers[i + 1](x, training)
            else:
                if self.is_layer_norm:
                    x = self.layer_layers[i](x)

            if self.is_dropout:
                x = self.dropout_layers[i](x)

        x = self.dense_layers[-1](x)

        x = self.reshape_layer(x)
        return x

    @tf.function
    def grad_call(self, x_tf, training):
        # # # returns the derivative of the network
        # # # important: x_tf needs to be a tensorflow object: eager tensor, constant, tensor, etc. (not numpy)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            output = self.call(x_tf, training)
        gradient = tape.gradient(output, x_tf)
        return gradient

    def jacobian_call(self, x_tf, training):
        print("SubNet: jacobian_call: retracing")
        if (not self.is_batch_norm) or (not training):
            return self.batch_jacobian_call(x_tf, training)

        # # # returns the jacobian of the network
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tf)  # (P, d)
            output = self.call(x_tf, training)  # (M, q)

        jacobian = tape.jacobian(output, x_tf)
        return jacobian  # (M, q, P, d)

    @tf.function(experimental_relax_shapes=True)
    def batch_jacobian_call(self, x_tf, training):
        # # # returns the jacobian of the network
        print("SubNet: batch_jacobian_call: retracing")
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tf)  # in R^d
            output = self.call(x_tf, training)  # in R^q
        jacobian = tape.batch_jacobian(output, x_tf, experimental_use_pfor=True)  # R^(qxd)
        # x_tf in R^(Mxd); output in R^(Mxq) -> jacobian in R^(Mxqxd)

        return jacobian

    @tf.function
    def hessian_call(self, x_tf, training):
        # # # returns the second derivative of the network
        # # # just as above: x_tf is assumed to be a tensorflow object
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            output = self.grad_call(x_tf, training)
        hessian = tape.jacobian(output, x_tf)
        hessian = tf.reduce_sum(hessian, -2)  # batches are independent
        return hessian

    @tf.function
    def batch_hessian_call(self, x_tf, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            grad = self.grad_call(x_tf, training)
        hessian = tape.batch_jacobian(grad, x_tf)
        return hessian
