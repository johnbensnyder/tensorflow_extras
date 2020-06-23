import tensorflow as tf
import tensorflow_addons as tfa

class RMSpropW(tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension, tf.keras.optimizers.RMSprop):
    def __init__(self, weight_decay, *args, **kwargs):
        super(RMSpropW, self).__init__(weight_decay, *args, **kwargs)
