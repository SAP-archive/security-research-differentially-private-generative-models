import tensorflow as tf
import numpy as np


from differential_privacy.dp_sgd.dp_optimizer.sanitizers.base import Sanitizer

class GroupedClipper(Sanitizer):

    def __init__(self, specials=None):
        i = 0


    def clip_grads(self,t,bound,name=None):

        assert bound > 0
        with tf.name_scope(values=[t, bound], name=name,
                           default_name="batch_clip_by_l2norm") as name:
            saved_shape = tf.shape(t)
        batch_size = tf.slice(saved_shape, [0], [1])

        # Add a small number to avoid divide by 0
        t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
        upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                                  tf.constant(1.0 / bound))
        # Add a small number to avoid divide by 0
        l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
        scale = tf.minimum(l2norm_inv, upper_bound_inv) * bound
        clipped_t = tf.matmul(tf.diag(scale), t2)
        clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
        return clipped_t


    def add_noise(self, t, sigma, name=None):

        """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

        Args:
          t: the input tensor.
          sigma: the stddev of the Gaussian noise.
          name: optional name.
        Returns:
          the noisy tensor.
        """

        with tf.name_scope(values=[t, sigma], name=name,
                           default_name="add_gaussian_noise") as name:
            noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
        return noisy_t