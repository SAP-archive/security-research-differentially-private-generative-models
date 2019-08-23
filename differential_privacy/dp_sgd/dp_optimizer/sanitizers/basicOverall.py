
import tensorflow as tf
import numpy as np


from differential_privacy.dp_sgd.dp_optimizer.sanitizers.base import Sanitizer

class BasicClipperOverall(Sanitizer):

    def __init__(self, upper_bound, l2norm_inv, specials=None):
        self.upper_bound = upper_bound
        self.l2norm_inv = l2norm_inv

    def clip_grads(self, t, name=None):
        """Clip an array of tensors by L2 norm.

        Shrink each dimension-0 slice of tensor (for matrix it is each row) such
        that the l2 norm is at most upper_bound. Here we clip each row as it
        corresponds to each example in the batch.

        Args:
            t: the input tensor.
            upper_bound: the upperbound of the L2 norm.
            name: optional name.
        Returns:
            the clipped tensor.
        """
        assert self.upper_bound > 0
        with tf.name_scope(values=[t, self.upper_bound], name=name,
                           default_name="batch_clip_by_l2norm") as name:
            saved_shape = tf.shape(t)
        batch_size = tf.slice(saved_shape, [0], [1])

        t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
        upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                                  tf.constant(1.0 / self.upper_bound))
        scale = tf.minimum(self.l2norm_inv, upper_bound_inv) * self.upper_bound
        clipped_t = tf.matmul(tf.diag(scale), t2)
        clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
        return clipped_t, upper_bound_inv

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