import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
from util import SharedLogDirichletInitializer, MaxEntInitializer


class TerpretProblem:
    """Tensorflow encoding of the 'Terpret problem'"""

    def __init__(self, opts):
        self.opts = opts
        self.build_graph()

    def build_graph(self):
        opts = self.opts

        # mu0 : T [1, 2]
        # This is the marginal for the variable x0 which is observed to be 0.
        self.mu0 = tf.constant([[1.0, 0.0]], dtype=tf.float32)

        # ms : T [k-1, 2]
        # These are the parameters we are trying to optimize for, that control the marginal distribution
        # of the xs in the chain.
        initializer = MaxEntInitializer(opts.k - 1, 2) if opts.max_ent else SharedLogDirichletInitializer(opts.alpha,
                                                                                                          opts.k - 1, 2)
        self.ms = tf.get_variable("ms", shape=[opts.k - 1, 2], initializer=initializer)

        # mus : T [k, 2]
        # These are the parameters for the distributions of the full xs chain.
        self.mus = tf.concat([self.mu0, tf.nn.softmax(self.ms)], axis=0)
        self.mus_stoch = tf.concat([self.mu0, tfp.distributions.RelaxedOneHotCategorical(opts.temp, self.ms).sample()],
                                   axis=0)

        self.mus_opt = self.mus_stoch if opts.stochastic else self.mus

        if not opts.mean_field:
            # Returns the probability that `xor(xs[i], xs[i+1%k])` equals `0`
            def soft_xor_p0(i):
                j = tf.mod(i + 1, opts.k)
                return (self.mus_opt[i, 0] * self.mus_opt[j, 0]) + (self.mus_opt[i, 1] * self.mus_opt[j, 1])

            # ys_eq_0 : T [k]
            self.ys_eq_0 = tf.map_fn(soft_xor_p0, tf.range(opts.k), dtype=tf.float32)
            self.entropy = tf.reduce_sum(-tf.log(tf.nn.softmax(self.ms)) * tf.nn.softmax(self.ms))
            self.loss = - tf.reduce_sum(tf.log(self.ys_eq_0)) + opts.entropy_weight * self.entropy
            self.update = tf.train.AdamOptimizer(learning_rate=opts.learning_rate).minimize(self.loss)
            self.other_mu = self.mus_stoch
        else:
            # calculate posterior updates assuming the XOR outputs are 0.
            def mf_update(i, current_q):
                i_left = tf.mod(i - 1, opts.k)
                i_right = tf.mod(i + 1, opts.k)
                avg = (self.mus_opt[i, :] + current_q[i_left, :] + current_q[i_right, :]) / 3.0
                return avg

            q_0 = self.mus_opt
            q_1 = tf.map_fn(lambda i: mf_update(i, q_0), tf.range(opts.k), dtype=tf.float32)

            self.target_mu = q_1[1:, :]
            self.other_mu = tf.concat([self.mu0, self.target_mu], axis=0)

            # then do cross entropy loss on posterior.
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_mu, logits=self.ms))
            self.update = tf.train.AdamOptimizer(learning_rate=opts.learning_rate).minimize(self.loss)
