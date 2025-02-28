import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class ChenSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    base class:
    https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L33-L60

    schedule from paper: https://arxiv.org/pdf/1909.11532.pdf
    """

    def __init__(self, initial_learning_rate, changing_learning_rate, lower_bound_learning_rate, total_epochs,
                 num_of_batches=None, staircase=False, name=None):
        super(ChenSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = changing_learning_rate
        self.lower_bound_learning_rate = lower_bound_learning_rate
        self.total_epochs = total_epochs
        self.num_of_batches = num_of_batches
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ChenSchedule") as name:
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_rate = math_ops.cast(self.decay_rate, dtype)
            total_epochs = math_ops.cast(self.total_epochs, dtype)

            # # # state reconstruct:
            if self.num_of_batches is not None:
                num_of_batches = math_ops.cast(self.num_of_batches, dtype)
                s = math_ops.cast(step * num_of_batches, dtype)
            else:
                s = math_ops.cast(step, dtype)

            if self.staircase:
                raise NotImplementedError
            power = math_ops.maximum(math_ops.minimum((s - total_epochs / 4) / (350 * total_epochs / 600), 1), 0)
            lr_updated = math_ops.multiply(initial_learning_rate, math_ops.pow(decay_rate, power), name=name)
            if self.lower_bound_learning_rate is not None:
                lr_updated = math_ops.maximum(lr_updated, self.lower_bound_learning_rate)
            return lr_updated

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
            "total_epochs": self.total_epochs,
            "num_of_batches": self.num_of_batches,
            "staircase": self.staircase,
            "name": self.name
        }

class BachouchSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    base class:
    https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L33-L60

    schedule from paper: https://arxiv.org/abs/1812.05916

    Idea is basically this:
    1) start the training with initial learning rate
    2) store the minimum validation loss over the training
    3) when the minimal validation loss rolls out of the last "monitoring_window" epochs adjust the learning rate to
    learning_rate = decay_rate * learning_rate

    STOPPING criteria:
    if learning_rate == lower_bound_learning_rate and dumping is requested
    """



    def __init__(self, initial_learning_rate, decay_rate, lower_bound_learning_rate, monitoring_window_width,
                 num_of_batches=None, staircase=False, name=None):
        super(BachouchSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.lower_bound_learning_rate = lower_bound_learning_rate
        self.monitoring_window_width = monitoring_window_width
        self.minimum_valid_loss_epoch_index = 0
        self.minimum_valid_loss = None
        self.current_epoch = None

        self.staircase = staircase
        self.name = "Bachouch_learning_rate_schedule"

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ChenSchedule") as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_rate = math_ops.cast(self.decay_rate, dtype)
            total_epochs = math_ops.cast(self.total_epochs, dtype)

            # # # state reconstruct:
            if self.num_of_batches is not None:
                num_of_batches = math_ops.cast(self.num_of_batches, dtype)
                s = math_ops.cast(step * num_of_batches, dtype)
            else:
                s = math_ops.cast(step, dtype)

            if self.staircase:
                raise NotImplementedError
            power = math_ops.maximum(math_ops.minimum((s - total_epochs / 4) / (350 * total_epochs / 600), 1), 0)
            lr_updated = math_ops.multiply(initial_learning_rate, math_ops.pow(decay_rate, power), name=name)
            if self.lower_bound_learning_rate is not None:
                lr_updated = math_ops.maximum(lr_updated, self.lower_bound_learning_rate)
            return lr_updated

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_rate": self.decay_rate,
            "total_epochs": self.total_epochs,
            "num_of_batches": self.num_of_batches,
            "staircase": self.staircase,
            "name": self.name
        }

class ChenScheduleKeras(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

  def __init__(self, schedule):
    super(ChenScheduleKeras, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))
