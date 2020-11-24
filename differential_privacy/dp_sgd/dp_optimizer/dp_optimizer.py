# (originally from https://github.com/tensorflow/models/tree/master/research/differential_privacy,
# possibly with some small edits by @corcra)

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# SPDX-FileCopyrightText: 2020 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

"""Differentially private optimizers.
"""
from __future__ import division

import tensorflow as tf

from differential_privacy.dp_sgd.dp_optimizer import utils

class DPGradientDescentOptimizer(tf.train.AdamOptimizer):
  """Differentially private gradient descent optimizer.
  """
  iteration = 0
  def __init__(self, learning_rate, eps_delta, sanitizer,
               sigma=None, use_locking=False, name="Adam",
               batches_per_lot=1):
    """Construct a differentially private gradient descent optimizer.

    The optimizer uses fixed privacy budget for each batch of training.

    Args:
      learning_rate: for GradientDescentOptimizer.
      eps_delta: EpsDelta pair for each epoch.
      sanitizer: for sanitizing the graident.
      sigma: noise sigma. If None, use eps_delta pair to compute sigma;
        otherwise use supplied sigma directly.
      use_locking: use locking.
      name: name for the object.
      batches_per_lot: Number of batches in a lot.
    """

    super(DPGradientDescentOptimizer, self).__init__(learning_rate, beta1=0.5, beta2=0.9, use_locking = use_locking, name = name)
    # Also, if needed, define the gradient accumulators
    self._batches_per_lot = batches_per_lot
    self._grad_accum_dict = {}
    if batches_per_lot > 1:
      self._batch_count = tf.Variable(1, dtype=tf.int32, trainable=False,
                                      name="batch_count")
      var_list = tf.trainable_variables()
      with tf.variable_scope("grad_acc_for"):
        for var in var_list:
          v_grad_accum = tf.Variable(tf.zeros_like(var),
                                     trainable=False,
                                     name=utils.GetTensorOpName(var))
          self._grad_accum_dict[var.name] = v_grad_accum

    self._eps_delta = eps_delta
    self._sanitizer = sanitizer
    self._sigma = sigma

  def compute_sanitized_gradients(self, loss, iteration,var_list=None,
                                  add_noise=True):
    """Compute the sanitized gradients.

    Args:
      loss: the loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """

    self._assert_valid_dtypes([loss])

    xs = [tf.convert_to_tensor(x) for x in var_list]
    # TODO check this change

    px_grads_byexample = tf.gradients(loss, xs)

    #loss_list = tf.unstack(loss, axis=0)
    #px_grads_byexample = [tf.gradients(l, xs) for l in loss_list]
    #px_grads = [[x[v] for x in px_grads_byexample] for v in range(len(xs))]
    px_grads = [px_grads_byexample[v] for v in range(len(xs))]

    #px_grads = tf.gradients(loss, xs)
    # add a dummy 0th dimension to reflect the fact that we have a batch size of 1...
  #  px_grads = [tf.expand_dims(x, 0) for x in px_grads]
#    px_grads = per_example_gradients.PerExampleGradients(loss, xs)

    sanitized, clipped, num_ex = self.call_sanitize_basic(px_grads, var_list)
    bound = []
    #sanitized, clipped = self._sanitizer.sanitize_overall(px_grads, var_list,self._eps_delta, sigma=self._sigma,
    #                                                   add_noise=add_noise,batches_per_lot=self._batches_per_lot)  # remove l2norm_inv to come back to clipping on each layer
    #sanitized, clipped = self.call_sanitize_grouped(px_grads, var_list,iteration)
    #sanitized, clipped, bound = self.call_sanitize_group(px_grads, var_list,add_noise,[[0],[2],[4],[6],[1],[3],[5],[7]])#[[0,2,4,6],[1,3,5,7]])#[[0,2,4],[6],[1],[3],[5],[7]])#
    return sanitized, clipped, px_grads, bound, num_ex

  def call_sanitize_group(self, px_grads, var_list, add_noise, groups): #sanitize with gradients divided in groups of layers (each group is a set of layers))
      sanitized_grads = [None] * len(var_list)
      clipped_grads = [None] * len(var_list)
      for group in groups:
        sigma_multiplier = 1  # 0.7#0.7
        bound_multiplier = 1 # 2#5
        if(group[0]%2!=0):
            sigma_multiplier = 1.3
            bound_multiplier = 0.5
        sanitized_grad, clipped_grad, bound = self._sanitizer.sanitize_overall([ px_grads[i] for i in group], [ var_list[i] for i in group],self._eps_delta, sigma=(self._sigma*sigma_multiplier), bound_multiplier = bound_multiplier,
                                                       add_noise=add_noise,batches_per_lot=self._batches_per_lot)
        for i in range(len(sanitized_grad)):
              sanitized_grads[group[i]] = sanitized_grad[i]
              clipped_grads[group[i]] = clipped_grad[i]
      return sanitized_grads,clipped_grads, bound

  def call_sanitize_grouped(self,px_grads, var_list,iteration):
      sanitized_grads = []
      clipped_grads = []
      index = 0
      for px_grad, v in zip(px_grads, var_list):
        mul_l2norm_bound = 1
        tensor_name = utils.GetTensorOpName(v)
        #if (tensor_name[-1] == 'b'):
        #    mul_l2norm_bound *= 1
        #mul_l2norm_bound *= int(tensor_name[14])
        privacy_multiplier = tf.add(tf.multiply(tf.mod(tf.add(iteration,tf.constant(index,tf.float32)), tf.constant(12.0,tf.float32)),tf.constant(0.05,tf.float32)),tf.constant(1.0))
        #tf.add(tf.subtract(iteration,iteration),tf.constant(1,tf.float32))
        curr_sigma = self._sigma * privacy_multiplier #* 0.4 * ((curr_iteration + index) % 6) #self._iteration#tf.constant(1.0,tf.float32)#self._iteration#* (5-int(tensor_name[14]))
        mul_l2norm_bound /= tf.multiply(privacy_multiplier,tf.constant(2.0,tf.float32))
        index += 1
        sanitized_grad, clipped_grad = self._sanitizer.sanitize_grouped(
          px_grad, self._eps_delta, sigma=curr_sigma,
          tensor_name=tensor_name, add_noise=True,
          num_examples=self._batches_per_lot * tf.slice(
              tf.shape(px_grad), [0], [1]), mul_l2norm_bound=mul_l2norm_bound)  # remove l2norm_inv to come back to clipping on each layer
        sanitized_grads.append(sanitized_grad)
        clipped_grads.append(clipped_grad)
      return sanitized_grads, clipped_grads

  def call_sanitize_basic(self,px_grads, var_list): #basic sanitizer with different parameters for bias weights
    sanitized_grads=[]
    clipped_grads = []
    for px_grad, v in zip(px_grads, var_list):
        tensor_name = utils.GetTensorOpName(v)
        if(tensor_name[-1]=='b'):# and tensor_name[14]=='4'):
            isBias = True
        else:
            isBias = False
        sanitized_grad, clipped_grad, num_ex = self._sanitizer.sanitize(
            px_grad, self._eps_delta, sigma=self._sigma,
            tensor_name=tensor_name, add_noise=True,
            num_examples=self._batches_per_lot * tf.slice(
            tf.shape(px_grad), [0], [1]),isBias=isBias) #remove l2norm_inv to come back to clipping on each layer
        sanitized_grads.append(sanitized_grad)
        clipped_grads.append(clipped_grad)
    return sanitized_grads, clipped_grads, num_ex

  def minimize(self, loss, iteration, global_step=None, var_list=None,
               name=None):
    """Minimize using sanitized gradients.

    This gets a var_list which is the list of trainable variables.
    For each var in var_list, we defined a grad_accumulator variable
    during init. When batches_per_lot > 1, we accumulate the gradient
    update in those. At the end of each lot, we apply the update back to
    the variable. This has the effect that for each lot we compute
    gradients at the point at the beginning of the lot, and then apply one
    update at the end of the lot. In other words, semantically, we are doing
    SGD with one lot being the equivalent of one usual batch of size
    batch_size * batches_per_lot.
    This allows us to simulate larger batches than our memory size would permit.

    The lr and the num_steps are in the lot world.

    Args:
      loss: the loss tensor.
      global_step: the optional global step.
      var_list: the optional variables.
      name: the optional name.
    Returns:
      the operation that runs one step of DP gradient descent.
    """

    # First validate the var_list

    if var_list is None:
      var_list = tf.trainable_variables()
    for var in var_list:
      if not isinstance(var, tf.Variable):
        raise TypeError("Argument is not a variable.Variable: %s" % var)

    # Modification: apply gradient once every batches_per_lot many steps.
    # This may lead to smaller error

    if self._batches_per_lot == 1:
      sanitized_grads, clipped_grads, gradient, bound, num_ex = self.compute_sanitized_gradients(
          loss,iteration, var_list=var_list)

      grads_and_vars = list(zip(sanitized_grads, var_list))

      self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])

      apply_grads = self.apply_gradients(grads_and_vars,
                                         global_step=global_step, name=name)
      return apply_grads, sanitized_grads, clipped_grads, gradient, bound, num_ex

    # Condition for deciding whether to accumulate the gradient
    # or actually apply it.
    # we use a private self_batch_count to keep track of number of batches.
    # global step will count number of lots processed.

    update_cond = tf.equal(tf.constant(0),
                           tf.mod(self._batch_count,
                                  tf.constant(self._batches_per_lot)))

    # Things to do for batches other than last of the lot.
    # Add non-noisy clipped grads to shadow variables.

    def non_last_in_lot_op(loss, var_list,iteration):
      """Ops to do for a typical batch.

      For a batch that is not the last one in the lot, we simply compute the
      sanitized gradients and apply them to the grad_acc variables.

      Args:
        loss: loss function tensor
        var_list: list of variables
      Returns:
        A tensorflow op to do the updates to the gradient accumulators
      """
      sanitized_grads = self.compute_sanitized_gradients(
          loss, iteration, var_list=var_list, add_noise=False)

      update_ops_list = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        update_ops_list.append(grad_acc_v.assign_add(grad))
      update_ops_list.append(self._batch_count.assign_add(1))
      return tf.group(*update_ops_list)

    # Things to do for last batch of a lot.
    # Add noisy clipped grads to accumulator.
    # Apply accumulated grads to vars.

    def last_in_lot_op(loss, var_list, global_step, iteration):
      """Ops to do for last batch in a lot.

      For the last batch in the lot, we first add the sanitized gradients to
      the gradient acc variables, and then apply these
      values over to the original variables (via an apply gradient)

      Args:
        loss: loss function tensor
        var_list: list of variables
        global_step: optional global step to be passed to apply_gradients
      Returns:
        A tensorflow op to push updates from shadow vars to real vars.
      """

      # We add noise in the last lot. This is why we need this code snippet
      # that looks almost identical to the non_last_op case here.
      sanitized_grads = self.compute_sanitized_gradients(
          loss, iteration, var_list=var_list, add_noise=True)

      normalized_grads = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        # To handle the lr difference per lot vs per batch, we divide the
        # update by number of batches per lot.
        normalized_grad = tf.div(grad_acc_v.assign_add(grad),
                                 tf.to_float(self._batches_per_lot))

        normalized_grads.append(normalized_grad)

      with tf.control_dependencies(normalized_grads):
        grads_and_vars = list(zip(normalized_grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars if g is not None])
        apply_san_grads = self.apply_gradients(grads_and_vars,
                                               global_step=global_step,
                                               name="apply_grads")

      # Now reset the accumulators to zero
      resets_list = []
      with tf.control_dependencies([apply_san_grads]):
        for _, acc in self._grad_accum_dict.items():
          reset = tf.assign(acc, tf.zeros_like(acc))
          resets_list.append(reset)
      resets_list.append(self._batch_count.assign_add(1))

      last_step_update = tf.group(*([apply_san_grads] + resets_list))
      return last_step_update
    # pylint: disable=g-long-lambda
    update_op = tf.cond(update_cond,
                        lambda: last_in_lot_op(
                            loss, var_list,
                            global_step),
                        lambda: non_last_in_lot_op(
                            loss, var_list, iteration))
    return tf.group(update_op)
