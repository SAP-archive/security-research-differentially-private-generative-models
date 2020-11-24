# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# SPDX-FileCopyrightText: 2020 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

"""Defines Sanitizer class for sanitizing tensors.

A sanitizer first limits the sensitivity of a tensor and then adds noise
to the tensor. The parameters are determined by the privacy_spending and the
other parameters. It also uses an accountant to keep track of the privacy
spending.
"""
from __future__ import division

import collections

import tensorflow as tf

from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.dp_sgd.dp_optimizer.sanitizers.basic import BasicClipper
from differential_privacy.dp_sgd.dp_optimizer.sanitizers.basicOverall import BasicClipperOverall
from differential_privacy.dp_sgd.dp_optimizer.sanitizers.grouped import GroupedClipper
import pdb

ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])


class AmortizedGaussianSanitizer(object):
  """Sanitizer with Gaussian noise and amoritzed privacy spending accounting.

  This sanitizes a tensor by first clipping the tensor, summing the tensor
  and then adding appropriate amount of noise. It also uses an amortized
  accountant to keep track of privacy spending.
  """

  def __init__(self, accountant, default_option, disc_params):
    """Construct an AmortizedGaussianSanitizer.

    Args:
      accountant: the privacy accountant. Expect an amortized one.
      default_option: the default ClipOptoin.
    """
    self.disc_parames = disc_params
    self._accountant = accountant
    self._default_option = default_option
    self._options = {}

  def set_option(self, tensor_name, option):
    """Set options for an individual tensor.

    Args:
      tensor_name: the name of the tensor.
      option: clip option.
    """

    self._options[tensor_name] = option

  # TODO ATTENTION, MY WORK
  def compute_overall_bound(self,t):
      saved_shape = tf.shape(t)
      batch_size = tf.slice(saved_shape, [0], [1])
      return tf.reshape(t, [1,-1])#[batch_size, [-1]]))

  def sanitize_overall(self, px_grads, var_list,eps_delta, option=ClipOption(None, None), num_examples=None, sigma=None, bound_multiplier = 1, add_noise=True,batches_per_lot=None):
    num_tot_examples = tf.zeros([1],dtype=tf.int32)
    sanitized_gradient = []
    clipped_gradients = []
    t_list = []
    weights_shapes = []
    weights_sizes = []
    linear_clipped_weights = []

    t2 = []

    for px_grad, v in zip(px_grads, var_list):
        t_list.append(tf.reshape(px_grad, tf.concat(axis=0, values=[tf.slice(tf.shape(px_grad), [0], [1]), [-1]])))#tf.reshape(px_grad, tf.concat(axis=0, values=[tf.slice(tf.shape(px_grad), [0], [1]), [-1]])))#self.compute_overall_bound(px_grad))

        l2norm_inv = tf.rsqrt(tf.reduce_sum(t_list[0] * t_list[0], [1]) + 0.000001)#tf.div(tf.constant(1.0),(tf.norm(t_list[0])+0.000001))
    t_overall = tf.concat(t_list, axis=0)
    #l2norm_inv = tf.div(tf.constant(1.0),tf.norm(t_overall))#tf.rsqrt(tf.reduce_sum(t_overall * t_overall, [1]) + 0.000001)#tf.div(tf.constant(1.0),tf.norm(px_grad))
    #t_overall = tf.reshape(t_overall, tf.concat(axis=0, values=[tf.slice(tf.shape(t_overall), [0], [1]), [-1]]))
    #l2norm_inv = tf.div(tf.constant(1.0),tf.norm(t_overall))#tf.rsqrt(tf.reduce_sum(t_overall * t_overall, [1]) + 0.000001)


    #    saved_shape = tf.shape(px_grad)
    #    batch_size = tf.slice(saved_shape, [0], [1])
    #    t2 = tf.reshape(px_grad, tf.concat(axis=0, values=[batch_size, [-1]]))
    # Add a small number to avoid divide by 0
    #l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)



    for px_grad, v in zip(px_grads, var_list):
      tensor_name = utils.GetTensorOpName(v)
      if sigma is None:
        # pylint: disable=unpacking-non-sequence
        eps, delta = eps_delta
        with tf.control_dependencies(
            [tf.Assert(tf.greater(eps, 0),
                       ["eps needs to be greater than 0"]),
             tf.Assert(tf.greater(delta, 0),
                       ["delta needs to be greater than 0"])]):
          # The following formula is taken from
          #   Dwork and Roth, The Algorithmic Foundations of Differential
          #   Privacy, Appendix A.
          #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
          sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

      l2norm_bound, clip = option
      if l2norm_bound is None:
        l2norm_bound, clip = self._default_option
        l2norm_bound *= bound_multiplier
        if ((v.name is not None) and
            (v.name in self._options)):
          l2norm_bound, clip = self._options[v.name]
      #clipper = GroupedClipper(self.disc_parames)
      clipper = BasicClipper(l2norm_bound)#BasicClipperOverall(l2norm_bound,l2norm_inv)#BasicClipper(l2norm_bound)#BasicClipperOverall(l2norm_bound,l2norm_inv)#BasicClipper(l2norm_bound)#BasicClipperOverall(l2norm_bound,l2norm_inv)#BasicClipper(l2norm_bound)#
      if clip:
        x, boundNew = clipper.clip_grads(px_grad)
        clipped_gradients.append(x)
        linear_clipped_weights.append(self.compute_overall_bound(x))
      num_examples_cur = tf.slice(tf.shape(x), [0], [1])
      if(x.shape.ndims>1):
        weights_sizes.append((x.shape[0]*x.shape[1]).value)
      else:
        weights_sizes.append(x.shape[0].value)
      weights_shapes.append(x.shape)
      num_tot_examples = tf.add(num_tot_examples,num_examples_cur)

      #saned_x = self.sanitize(
      #  px_grad, eps_delta, sigma=sigma,
      #  tensor_name=tensor_name, add_noise=add_noise,
      #  num_examples=batches_per_lot * tf.slice(
      #    tf.shape(px_grad), [0], [1]),
      #  isBias=False) #remove l2norm_inv to come back to clipping on each layer
      #sanitized_grads.append(sanitized_grad)
    #num_examples = tf.slice(tf.shape(t_overall), [0], [1])
    all_clipped_weights = tf.concat(linear_clipped_weights, axis=-1)
    privacy_accum_op = self._accountant.accumulate_privacy_spending(
    eps_delta, sigma, 200)
    with tf.control_dependencies([privacy_accum_op]):
      saned_x = clipper.add_noise(all_clipped_weights, sigma * l2norm_bound)

    splits = tf.split(saned_x,weights_sizes,1)
    for i,split in enumerate(splits):
      sanitized_gradient.append(tf.reshape(split,weights_shapes[i]))
    return sanitized_gradient, clipped_gradients, boundNew

  def sanitize_grouped(self, x, eps_delta, sigma=None,
               option=ClipOption(None, None), tensor_name=None,
               num_examples=None, add_noise=True, mul_l2norm_bound=None ):
    """Sanitize the given tensor.

       This santize a given tensor by first applying l2 norm clipping and then
       adding Gaussian noise. It calls the privacy accountant for updating the
       privacy spending.

       Args:
         x: the tensor to sanitize.
         eps_delta: a pair of eps, delta for (eps,delta)-DP. Use it to
           compute sigma if sigma is None.
         sigma: if sigma is not None, use sigma.
         option: a ClipOption which, if supplied, used for
           clipping and adding noise.
         tensor_name: the name of the tensor.
         num_examples: if None, use the number of "rows" of x.
         add_noise: if True, then add noise, else just clip.
       Returns:
         a pair of sanitized tensor and the operation to accumulate privacy
         spending.
       """

    if sigma is None:
      # pylint: disable=unpacking-non-sequence
      eps, delta = eps_delta
      with tf.control_dependencies(
              [tf.Assert(tf.greater(eps, 0),
                         ["eps needs to be greater than 0"]),
               tf.Assert(tf.greater(delta, 0),
                         ["delta needs to be greater than 0"])]):
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

    l2norm_bound, clip = option
    if l2norm_bound is None:
      l2norm_bound, clip = self._default_option

      if ((tensor_name is not None) and
              (tensor_name in self._options)):
        l2norm_bound, clip = self._options[tensor_name]
    l2norm_bound = mul_l2norm_bound * l2norm_bound
    clipper = BasicClipper(l2norm_bound)
    if clip:
      x = clipper.clip_grads(x)
    if add_noise:
      if num_examples is None:
        num_examples = tf.slice(tf.shape(x), [0], [1])
      privacy_accum_op = self._accountant.accumulate_privacy_spending(
        eps_delta, sigma, num_examples)
      with tf.control_dependencies([privacy_accum_op]):
        saned_x = clipper.add_noise(x, tf.multiply(sigma,l2norm_bound))
    else:
      saned_x = tf.reduce_sum(x, 0)
    return saned_x, x


  def sanitize(self, x, eps_delta, sigma=None,
               option=ClipOption(None, None), tensor_name=None,
               num_examples=None, add_noise=True, isBias=False):
    """Sanitize the given tensor.

    This santize a given tensor by first applying l2 norm clipping and then
    adding Gaussian noise. It calls the privacy accountant for updating the
    privacy spending.

    Args:
      x: the tensor to sanitize.
      eps_delta: a pair of eps, delta for (eps,delta)-DP. Use it to
        compute sigma if sigma is None.
      sigma: if sigma is not None, use sigma.
      option: a ClipOption which, if supplied, used for
        clipping and adding noise.
      tensor_name: the name of the tensor.
      num_examples: if None, use the number of "rows" of x.
      add_noise: if True, then add noise, else just clip.
    Returns:
      a pair of sanitized tensor and the operation to accumulate privacy
      spending.
    """

    if sigma is None:
      # pylint: disable=unpacking-non-sequence
      eps, delta = eps_delta
      with tf.control_dependencies(
          [tf.Assert(tf.greater(eps, 0),
                     ["eps needs to be greater than 0"]),
           tf.Assert(tf.greater(delta, 0),
                     ["delta needs to be greater than 0"])]):
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

    l2norm_bound, clip = option
    if l2norm_bound is None:
      l2norm_bound, clip = self._default_option

      if ((tensor_name is not None) and
          (tensor_name in self._options)):
        l2norm_bound, clip = self._options[tensor_name]
    #clipper = GroupedClipper(self.disc_parames)
    if (isBias):
        sigma *= 1.3#0.7
        l2norm_bound *= 0.5#2#5
    clipper = BasicClipper(l2norm_bound)
    if clip:
      x = clipper.clip_grads(x)
    if add_noise:
      if num_examples is None:
        num_examples = tf.slice(tf.shape(x), [0], [1])
      privacy_accum_op, q = self._accountant.accumulate_privacy_spending(
          eps_delta, sigma, num_examples)#TODO CHECK WHAT IT IS CORRECT num_examples) 200
      with tf.control_dependencies([privacy_accum_op]):
        saned_x = clipper.add_noise(x,sigma * l2norm_bound)
    else:
      saned_x = tf.reduce_sum(x, 0)
    return saned_x, x, q
