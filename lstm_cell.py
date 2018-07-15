#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2(c): Long-Short-Term-Memory
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("lstm_cell")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the LSTM equations are:

        """
        scope = scope or type(self).__name__
        h = state[0]
        c = state[1]
        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)

            # equations coming from http://colah.github.io/posts/2015-08-Understanding-LSTMs/

            W_i = tf.get_variable(name="W_i", shape=[self.input_size,2*self.state_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable(name="b_i", shape=[2*self.state_size], dtype=tf.float32,initializer=tf.constant_initializer(0))

            W_f = tf.get_variable(name="W_f", shape=[self.input_size,2*self.state_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable(name="b_f", shape=[2*self.state_size], dtype=tf.float32,initializer=tf.constant_initializer(0))

            W_o = tf.get_variable(name="W_o", shape=[self.input_size, 2*self.state_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable(name="b_o", shape=[2*self.state_size], dtype=tf.float32,initializer=tf.constant_initializer(0))

            i_t = tf.sigmoid(tf.matmul(tf.concat([h, inputs], 1), W_i) + b_i)
            f_t = tf.sigmoid(tf.matmul(tf.concat([h, inputs], 1), W_f) + b_f)
            c_hat_t = tf.nn.tanh(tf.matmul(tf.concat([h, inputs], W_c)))

            new_state = tf.matmul(f_t, c) + tf.matmul(i_t, c_hat_t)
            o_t = tf.sigmoid(tf.matmul(tf.concat([h, inputs], 1), W_o) + b_o)
            output = o_t*tf.nn.tanh(new_state)

        return output, new_state
