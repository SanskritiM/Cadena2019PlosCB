'''
Regularized LNP model for neural system identification

Author: Santiago Cadena
Last update: April 2019
'''

import numpy as np
import os
from scipy import stats
import tensorflow as tf
import hashlib
import inspect
import random
import tf_slim as slim
from tf_slim import losses
from tf_slim import layers
from cnn_sys_ident.utils import *
from cnn_sys_ident.base import Model

tf.compat.v1.disable_resource_variables()

class LNP(Model):

    def build(self, smooth_reg_weight, sparse_reg_weight):
        self.smooth_reg_weight = smooth_reg_weight
        self.sparse_reg_weight = sparse_reg_weight
        with self.graph.as_default():
            tmp = layers.convolution2d(self.images, self.data.num_neurons, self.data.px_x, 1, 'VALID',
                                                  activation_fn=tf.exp,
                                                  normalizer_fn=None,
                                                  #weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                  weights_initializer=tf.keras.initializers.GlorotUniform(),
                                                  weights_regularizer=lambda w: smoothness_regularizer_2d_lnp(w, smooth_reg_weight)+\
                                                  l1_regularizer(w, sparse_reg_weight),
                                                  biases_initializer=tf.constant_initializer(value=0),
                                                  scope='lnp')
            with tf.compat.v1.variable_scope('lnp', reuse=True):
                self.weights = tf.compat.v1.get_variable('weights')
                self.biases = tf.compat.v1.get_variable('biases')
            self.prediction = tf.squeeze(tmp, [1, 2])
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            self.total_loss = self.get_log_likelihood() + tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.initialize()

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.mse, self.prediction]