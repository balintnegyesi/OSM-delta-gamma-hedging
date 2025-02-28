import tensorflow as tf
import numpy as np
from lib.SubNet import SubNet

class BaseBackwardModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(BaseBackwardModel, self).__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

    def euler_forward(self, t, x, y, z, dw):
        if self.bsde.is_sigma_in_control:
            control = tf.einsum('ink,ik->in', self.bsde.ForwardDiffusion.sigma_process_tf(t, x), z)  # mtx * vec over batches
            y_next = y - self.bsde.f_tf(t, x, y, z) * self.bsde.delta_t + tf.reduce_sum(control * dw, 1, keepdims=True)
        else:
            y_next = y - self.bsde.f_tf(t, x, y, z) * self.bsde.delta_t + tf.reduce_sum(z * dw, 1, keepdims=True)
        return y_next

    def call(self, inputs, training):
        raise NotImplementedError


class OSMModelZn(BaseBackwardModel):
    def __init__(self, config, bsde):
        super(OSMModelZn, self).__init__(config, bsde)
        self.Zn = SubNet(config, [self.bsde.J, self.bsde.d])
        self.Gn = SubNet(config, [self.bsde.J, self.bsde.d, self.bsde.d])

    def call(self, x, training):
        raise NotImplementedError  # rather code this in the losses: allows for more flexibility

class OSMModelYn(BaseBackwardModel):
    def __init__(self, config, bsde):
        super(OSMModelYn, self).__init__(config, bsde)
        self.Yn = SubNet(config, self.bsde.J)

    def call(self, x, training):
        raise NotImplementedError  # rather code this in the losses: allows for more flexibility



class HureModeln(BaseBackwardModel):
    def __init__(self, config, bsde, is_Z_autodiff=False):
        super(HureModeln, self).__init__(config, bsde)
        self.Yn = SubNet(config, self.bsde.J)
        self.Zn = SubNet(config, [self.bsde.J, self.bsde.d])

    def call(self, x, training):
        raise NotImplementedError  # rather code this in the losses: allows for more flexibility
