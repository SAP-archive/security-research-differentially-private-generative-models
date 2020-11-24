# SPDX-FileCopyrightText: 2020 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import tflib as tflib
import tensorflow as tf

def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.leaky_relu(output)#relu(output)
    return output

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(logits),
        tf.shape(logits)
    )


def Generator(n_samples, real_data, disc_manager,diminput, DIM):
    noise = tf.random_normal([n_samples, diminput])
    output = LeakyReLULayer('Generator.1', diminput, DIM, noise)
    output = LeakyReLULayer('Generator.2', DIM, DIM, output)
    output = LeakyReLULayer('Generator.3', DIM, DIM, output)
    output = tflib.ops.linear.Linear('Generator.4', DIM, diminput, output)
    totIndex = 0
    outputs = []
    labelsLength = disc_manager.getLabelsLength()
    for length in labelsLength:
        if(length==1):
            outputs.append(tf.nn.sigmoid(output[:,totIndex:totIndex+length]))   #output[:,totIndex:totIndex+length])
        else:
            outputs.append(softmax(output[:,totIndex:totIndex+length]))
        totIndex += length
    output = tf.concat(outputs,axis=1)
    return output#tf.concat([output1, output2], 1)

def Generator_reconstruct(n_samples, real_data, disc_manager,diminput, DIM, noise):
    output = LeakyReLULayer('Generator.1', diminput, DIM, noise)
    output = LeakyReLULayer('Generator.2', DIM, DIM, output)
    output = LeakyReLULayer('Generator.3', DIM, DIM, output)
    output = tflib.ops.linear.Linear('Generator.4', DIM, diminput, output)
    totIndex = 0
    outputs = []
    labelsLength = disc_manager.getLabelsLength()
    for length in labelsLength:
        if(length==1):
            outputs.append(tf.nn.sigmoid(output[:,totIndex:totIndex+length]))   #output[:,totIndex:totIndex+length])
        else:
            outputs.append(softmax(output[:,totIndex:totIndex+length]))
        totIndex += length
    output = tf.concat(outputs,axis=1)
    return output#tf.concat([output1, output2], 1)

def Discriminator(inputs,diminput, DIM):
    output = LeakyReLULayer('Discriminator.1', diminput, DIM, inputs)
    output = LeakyReLULayer('Discriminator.2', DIM, DIM, output)
    output = LeakyReLULayer('Discriminator.3', DIM, DIM, output)
    output = tflib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

def Discriminator_bb(inputs,diminput, DIM):
    output = LeakyReLULayer('Bbdiscriminator.1', diminput, DIM, inputs)
    output = LeakyReLULayer('Bbdiscriminator.2', DIM, DIM, output)
    output = LeakyReLULayer('Bbdiscriminator.3', DIM, DIM, output)
    output = tflib.ops.linear.Linear('Bbdiscriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

def GeneratorRNN(n_samples, diminput, DIM, seq_length, hidden_units,num_generated_features,reuse=False):
    with tf.variable_scope("generator") as scope:
        #if reuse:
        #    scope.reuse_variables()
        lstm_initializer = None
        bias_start = 1.0
        #W_out_G_initializer = tf.truncated_normal_initializer()
        #b_out_G_initializer = tf.truncated_normal_initializer()
        #W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units, num_generated_features], initializer=W_out_G_initializer)
        #b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)

        noise = tf.random_normal([n_samples, seq_length, 1])
        cell = LSTMCell(num_units=hidden_units, state_is_tuple=True, bias_start=bias_start, initializer=lstm_initializer)
        #cell = LSTMCell(num_units=hidden_units, state_is_tuple=True)
        initial_state = cell.zero_state(96, tf.float32)

        #cell = LSTMCell(num_units=hidden_units,
        #                state_is_tuple=True
        #                )
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * n_samples,
            inputs=noise,
            )       #initial_state=initial_state, time_major=True

        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units])
        logits_2d = LeakyReLULayer('LSTM.1', hidden_units, 1, rnn_outputs_2d)
        #logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        #output_2d = tf.multiply(tf.nn.tanh(logits_2d), scale_out_G)
        output_2d = tf.nn.sigmoid(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

def DiscriminatorRNN(inputs,diminput, DIM,  reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        output = tf.reshape(inputs,[-1,diminput])
        output = LeakyReLULayer('Discriminator.1', diminput, DIM, output)
        output = LeakyReLULayer('Discriminator.2', DIM, DIM, output)
        output = LeakyReLULayer('Discriminator.3', DIM, DIM, output)
        output = tflib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
        return tf.reshape(output, [-1])
        #W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units, 1],
        #                          initializer=tf.truncated_normal_initializer())
        #b_out_D = tf.get_variable(name='b_out_D', shape=1,
        #                          initializer=tf.truncated_normal_initializer())
        #cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units,
        #                               state_is_tuple=True,reuse=reuse)
        #rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        #    cell=cell,
        #    dtype=tf.float32,
        #    inputs=inputs)
        #logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
        #output = tf.nn.sigmoid(logits)
    #return output, logits