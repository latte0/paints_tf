from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils.utility import *


class pix2pixmodel(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, output_size=256,
                 gf_dim=64, df_dim=64,
                 input_c_dim=3, output_c_dim=3,
                 wasserstein=False,
                 SELU=False):

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn = [0 for i in range(4)]
        self.g_bn_e = [0 for i in range(9)]
        self.g_bn_d = [0 for i in range(8)]
        for i in range(1,4):
            print i
            self.d_bn[i] = (batch_norm(name='d_bn' + str(i)) )

        for i in range(2,9):
            self.g_bn_e[i] = ( batch_norm(name='g_bn_e'+  str(i)) )

        for i in range(1,8):
            self.g_bn_d[i] = ( batch_norm(name='g_bn_d' + str(i)))

        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_size = output_size

        self.SELU = SELU
        self.wasserstein = wasserstein


    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            def lrelu_discriminator():
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn[1](conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn[2](conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn[3](conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                return h4

            def selu_discriminator():
                h0 = selu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = selu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
                h2 = selu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
                h3 = selu(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                return 0

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h4 = None

            if(self.SELU):
                h4 = selu_discriminator()
            else:
                h4 = lrelu_discriminator()

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, sampler=False, y=None):
        with tf.variable_scope("generator") as scope:

            layer_specs_deconv = [
                self.gf_dim * 2, # encoder_2: [batch, 128, 128, gf_dim] => [batch, 64, 64, gf_dim * 2]
                self.gf_dim * 4, # encoder_3: [batch, 64, 64, gf_dim * 2] => [batch, 32, 32, gf_dim * 4]
                self.gf_dim * 8, # encoder_4: [batch, 32, 32, gf_dim * 4] => [batch, 16, 16, gf_dim * 8]
                self.gf_dim * 8, # encoder_5: [batch, 16, 16, gf_dim * 8] => [batch, 8, 8, gf_dim * 8]
                self.gf_dim * 8, # encoder_6: [batch, 8, 8, gf_dim * 8] => [batch, 4, 4, gf_dim * 8]
                self.gf_dim * 8, # encoder_7: [batch, 4, 4, gf_dim * 8] => [batch, 2, 2, gf_dim * 8]
                self.gf_dim * 8, # encoder_8: [batch, 2, 2, gf_dim * 8] => [batch, 1, 1, gf_dim * 8]
            ]



            layer_specs_conv = [
       	        (self.gf_dim * 8, 0.5),   # decoder_8: [batch, 1, 1, gf_dim * 8] => [batch, 2, 2, gf_dim * 8 * 2]
                (self.gf_dim * 8, 0.5),   # decoder_7: [batch, 2, 2, gf_dim * 8 * 2] => [batch, 4, 4, gf_dim * 8 * 2]
                (self.gf_dim * 8, 0.5),   # decoder_6: [batch, 4, 4, gf_dim * 8 * 2] => [batch, 8, 8, gf_dim * 8 * 2]
                (self.gf_dim * 8, 0.0),   # decoder_5: [batch, 8, 8, gf_dim * 8 * 2] => [batch, 16, 16, gf_dim * 8 * 2]
                (self.gf_dim * 4, 0.0),   # decoder_4: [batch, 16, 16, gf_dim * 8 * 2] => [batch, 32, 32, gf_dim * 4 * 2]
                (self.gf_dim * 2, 0.0),   # decoder_3: [batch, 32, 32, gf_dim * 4 * 2] => [batch, 64, 64, gf_dim * 2 * 2]
                (self.gf_dim, 0.0),       # decoder_2: [batch, 64, 64, gf_dim * 2 * 2] => [batch, 128, 128, gf_dim * 2]
            ]


            if sampler:
                scope.reuse_variables()

            def generator_impl():
                self.e = [0 for i in range(9)]
                self.d = [0 for i in range(9)]
                self.d_w = [0 for i in range(9)]
                self.d_b = [0 for i in range(9)]

                self.e[1] = conv2d(image, self.gf_dim, name= 'g_e1_conv')

                for i in range(2,9):
                    self.e[i] = self.g_bn_e[i](conv2d(lrelu(self.e[i-1]), layer_specs_deconv[i-2], name='g_e'+ str(i) + '_conv' ))


                s = self.output_size
                s = int(s/128)

                self.d[1], self.d_w[1], self.d_b[1] = deconv2d(tf.nn.relu(self.e[8]),
                    [self.batch_size, s, s, layer_specs_conv[0][0] ], name='g_d1', with_w=True)

                self.d[1] = tf.nn.dropout(self.g_bn_d[1](self.d[1]), 0.5)
                self.d[1] = tf.concat([self.d[1], self.e[7]], 3)


                for i in range(2,8):
                    print(i)
                    s = int( self.output_size/(2**(8-i)) )

                    self.d[i], self.d_w[i], self.d_b[i] = deconv2d(tf.nn.relu(self.d[i-1]),
                          [self.batch_size, s, s, layer_specs_conv[i-1][0] ], name='g_d' + str(i), with_w=True)

                    if( layer_specs_conv[i-1][1] > 0.0 ): self.d[i] = tf.nn.dropout(self.g_bn_d[i](self.d[i]), 0.5)
                    else: self.d[i] = self.g_bn_d[i](self.d[i])

                    self.d[i] = tf.concat([self.d[i], self.e[8-i]], 3)


                s = int(self.output_size)

                self.d[8], self.d_w[8], self.d_b[8] = deconv2d(tf.nn.relu(self.d[7]),
                  [self.batch_size, s , s, self.output_c_dim], name='g_d8', with_w=True)

                return self.d[8], self.d_w[8], self.d_b[8]


            ret, _, _ = generator_impl()

            return tf.nn.tanh(ret)
